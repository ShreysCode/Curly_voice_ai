import json
from pathlib import Path
from typing import Any


class KnowledgeBase:

    def __init__(self, path: str | Path):

        self.path = Path(path)

        with self.path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.data: dict[str, Any] = json.load(file)

    def get_context(
        self,
        topic: str | None = None
    ) -> str:

        if topic == "organization":

            data = {
                "organization":
                    self.data.get(
                        "organization",
                        {}
                    )
            }

        elif topic == "lab":

            data = {
                "lab":
                    self.data.get(
                        "lab",
                        {}
                    )
            }

        else:

            data = self.data

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )
    
    def get_lab_field(self, field: str):
        lab = self.data.get("lab", {})
        return lab.get(field)
    
    def get_lab(self) -> dict:
        return self.data.get("lab", {})
    
    def find_faq(
    self,
    question: str
) -> str | None:

        question = question.lower().strip()

        for item in self.data.get(
        "faq",
        []
    ):

         faq_question = item[
            "question"
        ].lower()

         if question == faq_question:
            return item["answer"]

        return None