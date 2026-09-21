# Kazakhstan eGov RAG

A LangServe backend and Streamlit frontend for questions about Kazakhstan e-government services. It
loads the Russian QA dataset from `data/egov_ru.xlsx`, retrieves supporting
records locally, and uses Groq-hosted Qwen to answer only from that evidence.

## Architecture

The HTTP, application, retrieval, and agent layers are independent:

1. `api/app.py` mounts the LangGraph runnable at the API root with LangServe.
2. `api/runnable.py` constructs and types the question-answering runnable.
3. `retrieval/loader.py` uses pandas and Pydantic to validate Excel records, then
   consolidates duplicate services while retaining distinct `for_embedding` values
   as query aliases.
4. `retrieval/chroma.py` stores query aliases and canonical documents in separate
   local Chroma collections. It runs cosine and BM25 search over aliases, combines
   them with reciprocal-rank fusion (RRF), and collapses results by canonical service.
5. `retrieval/reranker.py` uses the local multilingual cross-encoder under
   `models/` to reorder canonical candidates and remove candidates below its
   relevance threshold.
6. `agent/graph.py` runs a LangGraph flow: prepare the retrieval query, retrieve,
   check evidence, answer or refuse, and finalize citations. LLM query
   summarization is optional and disabled by default to avoid a second chat-model
   inference on every request.
7. `agent/citations.py` validates model citations and deterministically appends
   cited document names and deduplicated `egov_link` values. Other URL fields
   are intentionally not exposed.
8. `serve.py` renders the structured answer, expandable sources, and an
   expandable useful-links block.

Embeddings, indexing, retrieval, and reranking stay local. For each answer, the
user question and the selected evidence records are sent to Groq. The complete
workbook and Chroma index are never uploaded.

## Setup

Python 3.12+, Ollama, and a Groq API key are required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b
```

Put the Groq API key in the untracked `.env` file as `GROQ_KEY`. The application
loads it into the process environment at startup. Never add the key to
`config.yaml`, source code, tests, or version control.

Groq HTTPS connections use the operating system certificate store. If a
corporate proxy uses a private CA that is not installed system-wide, export its
public certificate chain as a PEM bundle and set `GROQ_CA_BUNDLE` in `.env` to
that file's absolute path. TLS certificate verification is never disabled.

Place the QA workbook at `data/egov_ru.xlsx` and the unpacked reranker at
`models/mmarco-mMiniLMv2-L12-h384-v1`. Both directories are ignored by Git
except for their `.gitkeep` placeholders. The reranker is loaded in offline-only
mode and is never downloaded by the application. To run without it, set
`reranker.enabled: false` before starting the server.

All runtime settings live in `config.yaml`.

## Build the semantic index

```powershell
python create_index.py
```

This recreates the `egov_services` query-alias collection and its companion
`egov_services_documents` canonical-document collection under `data/chroma`.
The directory is ignored by Git. Indexing prints alias-embedding progress after
every Ollama batch. Search and question answering require both collections. An
interrupted or inconsistent build will not be used for answers.

## Run the app

```powershell
streamlit run serve.py
```

Streamlit automatically starts the local LangServe backend and warms up the
configured Ollama embedding model. If a backend is already running at the
configured address, the frontend reuses it.

For backend debugging, it can still be started separately:

```powershell
python backend.py
```

Ask a question with LangServe's `POST /invoke` endpoint:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/invoke `
  -ContentType "application/json" `
  -Body '{"input":{"question":"Как заменить удостоверение личности после смены фамилии?"}}'
```

The `output` field contains only the agent's final answer. A grounded response
contains inline citations, a source list built from the cited records, and an
optional deduplicated list of useful links:

```text
Подать заявление можно через портал. [1]

Источники:

[1] - Процедура регистрации брака

Полезные ссылки:

- https://egov.kz/...
```

Retrieval hits, grounding state, and LangGraph intermediate state remain
internal.

LangServe also exposes `/batch`, `/stream`, `/stream_log`, input/output schema,
and playground endpoints. Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`.

The frontend uses `POST /ask/invoke`, which returns the answer body, cited
source metadata, deduplicated `egov_link` values, and grounding status as
separate fields. The original `POST /invoke` text response remains available.

The backend address and frontend request timeout are configured under
`frontend` in `config.yaml`.

At backend startup, the configured Ollama embedding model is warmed up and
retained for five minutes. Set `server.warmup_embedding: false` to disable this
behavior. Chat generation is handled by Groq and requires no local warm-up.

## Configuration

Edit `config.yaml` to change workbook and Chroma paths, the local Ollama
embedding model, Groq chat model, indexing
batch size, retrieval/RRF settings, alias candidate expansion, the minimum
semantic-similarity threshold, reranker model and limits, summarization and
generation limits, reasoning mode, or server host and port. Hidden model
reasoning is disabled by default to keep responses fast and focused. The local
embedding model uses a five-minute keep-alive. Set `summarization.enabled: true` only when
evaluation shows that direct user questions do not retrieve reliably enough.

The backend logs durations for local Ollama embedding, Groq chat requests, query
preparation, retrieval, answer generation, and embedding warm-up. These entries
use the prefix `latency stage=`.

`retrieval.min_semantic_similarity` is applied to every candidate before the
final `top_k` results are selected and again before context generation. The
default is `0.60`; lower it only after evaluating rejected valid questions.

RRF supplies up to `reranker.candidate_k` canonical documents after the semantic
gate. The reranker scores those documents against the original user question,
orders them by direct relevance, applies `reranker.min_score`, and returns the
configured retrieval `top_k`. The default `min_score` is `0.0`, so reranking
initially changes ordering without rejecting additional documents. Calibrate it
on reviewed questions before raising it. If local reranking fails, the retriever
returns no unchecked evidence and the agent uses its safe insufficiency response.
Reranking is query-time only, so enabling or changing it does not require an
index rebuild.

## Langfuse monitoring

Langfuse tracing is optional and disabled by default. To enable it, copy the
variables from `.env.example` into your environment, provide credentials for
your Langfuse Cloud or self-hosted project, and set `langfuse.enabled: true` in
`config.yaml`.

```powershell
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
python backend.py
```

Each `/invoke` run is recorded under the configured `langfuse.run_name`, with
the LangGraph node hierarchy, timings, inputs, outputs, and errors. When using
Langfuse Cloud, question and answer trace data leave the local machine; use a
self-hosted endpoint or leave monitoring disabled if that is not acceptable.

Rebuild the index whenever the workbook, canonicalization logic, or embedding
model changes. Both collections store the source fingerprint and embedding
model name. Indexes created before the query-alias layout are intentionally
rejected and must be rebuilt with `python create_index.py`.

## Tests

```powershell
python -m pytest
```

Tests use an ephemeral Chroma client and fake embeddings. They do not call
Ollama or any network service.

## Evaluation set

`evals/rag_eval_30.jsonl` contains 30 answerable Russian questions mapped to
canonical document IDs, source record IDs, indexed reference answers, service
names, and expected eGov URLs. The questions are manually paraphrased and are
validated not to exactly duplicate any stored query alias.

Regenerate the reference fields from the current canonical Chroma index with:

```powershell
python evals/build_eval_set.py
```
