# Goal

The goal is to build a RAG-agent that uses a QA-dataset to answer questions about government services of Kazakhstan

# Technical Stack

- Python version: Python 3.12
- Data processing: Pandas
- Agent architecture: LangGraph / LangChain
- Vector Database: ChromaDB
- LLM API: local ollama
- Backend: Langserve

# Entry points

Project should implement the following entry Python scripts

- create_index.py rebuilds index
- backend.py serves the LangServe API locally
- serve.py serves the Streamlit app locally
- eval.py evaluate the retrieval

# Agent Architecture

- Input: User question
- Summarize query: Make a short summary of a user question
- Retrieval: Retrieve relevant chunks from the database based on summarized query
- Answer: Answer user question based on retrieved chunks

# Retrieval

- Retrieval is based on QA-dataset from Kazakhstan OpenData project
- Retrieval should be hybrid: BM25 + Semantic search
- Use ChromaDB as a vector database
- Use local reranker. The reranker model is stored in models/mmarco-mMiniLMv2-L12-h384-v1

# Models

- For embeddings, use local qwen3-embedding:0.6b though ollama
- For intent detection and answering, use local qwen3:4b
- Reranker: local mMiniLMv2-L12-h384-v1

# Answer requirements

- Answers must be grounded in retrieved evidence.
- Do not invent missing requirements, documents, deadlines, fees, or procedures.
- If retrieved evidence is insufficient, explicitly state that the available data is insufficient.
- Answers are in russian language

## Citations

- Answer should cite source documents from retrieval as [1], [2]
- Answer should contain a citation. The cited documents should be referenced by name field.

## URLs

- Some records in the retrieval dataset contain URL links to the relevant government pages
- Append a block 'Полезные ссылки' with links to a final answer after the citation list
- Retrieval records may contain duplicate links. Deduplicate first before adding to the final answer

Example:

Вопрос: Как зарегистрировать брак

Ответ:

Подать заявку на регистрацию брака можно онлайн или при обращении в ЦОН. [1]

Источники:
[1] - Процедура регистрации брака

Полезные ссылки:
- https://gov.kz/...

# Evaluation

- Entry-point eval.py should run retrieval evaluation
- Retrieval quality is measured on a JSON dataset of 30 relevant examples
- Metrics: Precision@3, Recall@3, MRR, Hit@1, Latency

# Frontend

- The app should have a minimalistic frontend developed in Python streamlit

## Design

- White background with large search bar. 
- Display loading spinner while use waits for the response.
- List sources as expandable elements after the text answer
- Source element has a header with source title and contains expandable answer text
- For the frontend, citation numbers should be removed from text
- Reference: Google main page

# Deployment

- App is deployed with docker compose
- Persist all the downloads in volumes
- On first build the app should:
    - Pull ollama image
    - Download the required ollama models.
    - Download the dataset from http://magda-minio-web.data.gov.kz/magda-datasets/egov_ru.xlsx
    - Rebuild Chroma index from the data
    - Start backend in a separate container before serving a web app
