from __future__ import annotations

import json
from pathlib import Path
import sys

import chromadb
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from govkz_rag.config import Settings


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str
    category: str
    question: str
    relevant_document_id: int = Field(gt=0)


CASES = (
    EvalCase(eval_id="egov-001", category="Акты гражданского состояния", question="Куда подать заявление, чтобы официально зарегистрировать брак через интернет?", relevant_document_id=4555),
    EvalCase(eval_id="egov-002", category="Акты гражданского состояния", question="Можно ли через eGov оформить свидетельство о рождении новорожденного и что делать после заявки?", relevant_document_id=1357),
    EvalCase(eval_id="egov-003", category="Акты гражданского состояния", question="Ребёнок появился на свет за пределами Казахстана. Как оформить его рождение через консульство?", relevant_document_id=1852),
    EvalCase(eval_id="egov-004", category="Документы", question="Куда обращаться для замены паспорта из-за порчи или изменения персональных данных?", relevant_document_id=1165),
    EvalCase(eval_id="egov-005", category="Документы", question="Как временно находящемуся в Казахстане иностранцу оформить ИИН и сколько это занимает?", relevant_document_id=959),
    EvalCase(eval_id="egov-006", category="Жильё и регистрация", question="Как собственнику онлайн снять жильца с регистрации по адресу?", relevant_document_id=1222),
    EvalCase(eval_id="egov-007", category="Недвижимость и земля", question="Где физлицу получить форму 2 с правами, обременениями и характеристиками квартиры?", relevant_document_id=956),
    EvalCase(eval_id="egov-008", category="Недвижимость и земля", question="Потерян кадастровый паспорт квартиры. Как заказать повторный экземпляр?", relevant_document_id=1766),
    EvalCase(eval_id="egov-009", category="Недвижимость и земля", question="Как подать заявку на 10 соток под ИЖС и проверить место в очереди?", relevant_document_id=780),
    EvalCase(eval_id="egov-010", category="Недвижимость и земля", question="Как заказать акт с кадастровой стоимостью принадлежащего мне земельного участка?", relevant_document_id=2009),
    EvalCase(eval_id="egov-011", category="Транспорт", question="Можно ли подать заявление на обмен водительских прав онлайн и где затем их забрать?", relevant_document_id=1554),
    EvalCase(eval_id="egov-012", category="Транспорт", question="Как восстановить утраченный государственный номер автомобиля?", relevant_document_id=1466),
    EvalCase(eval_id="egov-013", category="Транспорт", question="В базе неверно указаны сведения о моей машине. Как отправить запрос на исправление?", relevant_document_id=1168),
    EvalCase(eval_id="egov-014", category="Справки", question="Как получить электронную справку о наличии или отсутствии судимости для трудоустройства?", relevant_document_id=1634),
    EvalCase(eval_id="egov-015", category="Справки", question="Может ли работодатель запросить справку о несудимости на сотрудника и что должен подтвердить сотрудник?", relevant_document_id=1305),
    EvalCase(eval_id="egov-016", category="Воинская служба", question="Как оформить отсрочку от срочной воинской службы через eGov?", relevant_document_id=1998),
    EvalCase(eval_id="egov-017", category="Воинская служба", question="Как призывнику подать заявление о постановке на воинский учет и сколько ждать?", relevant_document_id=915),
    EvalCase(eval_id="egov-018", category="Семья и опека", question="Где онлайн заказать документ, подтверждающий, что я являюсь опекуном?", relevant_document_id=756),
    EvalCase(eval_id="egov-019", category="Социальная поддержка", question="Можно ли подать на установление инвалидности через интернет и где появится решение?", relevant_document_id=1423),
    EvalCase(eval_id="egov-020", category="Социальная поддержка", question="Как онлайн оформить компенсацию расходов на коммунальные услуги по месту прописки?", relevant_document_id=2076),
    EvalCase(eval_id="egov-021", category="Социальная поддержка", question="Как малообеспеченной семье подать заявление на адресную социальную помощь?", relevant_document_id=1484),
    EvalCase(eval_id="egov-022", category="Пенсии", question="Куда подать документы для назначения пенсии по возрасту и какой срок рассмотрения?", relevant_document_id=1428),
    EvalCase(eval_id="egov-023", category="Образование", question="Как поставить ребёнка младше шести лет в очередь в государственный детский сад?", relevant_document_id=1212),
    EvalCase(eval_id="egov-024", category="Здравоохранение", question="Как сменить поликлинику через eGov и когда придёт результ?", relevant_document_id=3855),
    EvalCase(eval_id="egov-025", category="Разрешения", question="Как получить удостоверение охотника через портал и сколько стоит услуга?", relevant_document_id=1694),
    EvalCase(eval_id="egov-026", category="Финансы", question="Где скачать свой кредитный отчёт перед подачей заявки на заем?", relevant_document_id=2094),
    EvalCase(eval_id="egov-027", category="Налоги", question="Можно ли узнать и погасить налоговую задолженность без ЭЦП?", relevant_document_id=1334),
    EvalCase(eval_id="egov-028", category="Бизнес", question="Как зарегистрировать ТОО вместе с открытием банковского счёта и страхованием работников?", relevant_document_id=1607),
    EvalCase(eval_id="egov-029", category="Бизнес", question="Как зарегистрировать общественный фонд или другую некоммерческую организацию онлайн?", relevant_document_id=779),
    EvalCase(eval_id="egov-030", category="Образование", question="Какие документы подать для участия ребёнка в конкурсе на грант «Өркен» для обучения в НИШ?", relevant_document_id=1483),
)


def build_eval_set(
    output_path: Path,
    config_path: Path = PROJECT_ROOT / "config.yaml",
) -> None:
    settings = Settings.from_yaml(config_path)
    client = chromadb.PersistentClient(path=str(settings.retrieval.chroma.path))
    collection = client.get_collection(
        f"{settings.retrieval.chroma.collection}_documents",
        embedding_function=None,
    )
    requested_ids = [str(case.relevant_document_id) for case in CASES]
    stored = collection.get(ids=requested_ids, include=["metadatas"])
    records = dict(zip(stored["ids"], stored["metadatas"], strict=True))
    missing = set(requested_ids).difference(records)
    if missing:
        raise RuntimeError(f"Missing canonical documents in index: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for case in CASES:
        metadata = records[str(case.relevant_document_id)]
        aliases = json.loads(str(metadata["alternative_queries"]))
        if case.question.casefold() in {str(alias).casefold() for alias in aliases}:
            raise RuntimeError(f"Eval question duplicates an indexed alias: {case.eval_id}")
        rows.append(
            {
                "eval_id": case.eval_id,
                "category": case.category,
                "question": case.question,
                "relevant_document_ids": [case.relevant_document_id],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_eval_set(PROJECT_ROOT / "evals/rag_eval_30.jsonl")
