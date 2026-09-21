from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://magda-minio-web.data.gov.kz/magda-datasets/egov_ru.xlsx"
RERANKER_REPOSITORY = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANKER_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _download_dataset(url: str, target: Path) -> None:
    from govkz_rag.retrieval.loader import load_documents

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        documents = load_documents(target)
        print(f"Dataset is ready: {len(documents)} canonical documents", flush=True)
        return

    temporary = target.with_name(f".{target.stem}.download.xlsx")
    print(f"Downloading dataset from {url}", flush=True)
    try:
        with httpx.Client(follow_redirects=True, timeout=300.0) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
        documents = load_documents(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Dataset downloaded: {len(documents)} canonical documents", flush=True)


def _reranker_is_ready(target: Path) -> bool:
    return target.is_dir() and all((target / name).is_file() for name in RERANKER_FILES)


def _download_reranker(target: Path) -> None:
    if _reranker_is_ready(target):
        print(f"Reranker is ready: {target}", flush=True)
        return
    if target.exists():
        raise RuntimeError(
            f"Reranker directory is incomplete: {target}. Remove the model volume "
            "and start Docker Compose again."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading reranker {RERANKER_REPOSITORY}", flush=True)
    with tempfile.TemporaryDirectory(
        prefix="reranker-",
        dir=target.parent,
    ) as temporary_root:
        temporary = Path(temporary_root) / target.name
        snapshot_download(
            repo_id=RERANKER_REPOSITORY,
            revision=RERANKER_REVISION,
            local_dir=temporary,
            allow_patterns=list(RERANKER_FILES),
        )
        if not _reranker_is_ready(temporary):
            raise RuntimeError("Downloaded reranker is missing required files")
        temporary.replace(target)
    print(f"Reranker downloaded: {target}", flush=True)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from govkz_rag.config import Settings

    settings = Settings.from_yaml(PROJECT_ROOT / "config.yaml")
    dataset_url = os.getenv("DATASET_URL", DATASET_URL)
    _download_dataset(dataset_url, settings.retrieval.data.workbook)
    if settings.retrieval.reranker.enabled:
        _download_reranker(settings.retrieval.reranker.model)


if __name__ == "__main__":
    main()
