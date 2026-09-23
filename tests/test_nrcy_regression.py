import asyncio
import json
from pathlib import Path

from app.core.curly import Curly
from app.knowledge.info_repository import InfoRepository


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "app" / "knowledge" / "data.json"
REPO_PATH = ROOT / "app" / "knowledge" / "info_repo.json"


class ExplodingLLM:
    async def generate(self, *args, **kwargs):
        raise AssertionError("Known person/role query must not reach the LLM")

    async def generate_stream(self, *args, **kwargs):
        raise AssertionError("Known person/role query must not reach the LLM")
        yield ""


class Knowledge:
    def __init__(self):
        with DATA_PATH.open("r", encoding="utf-8") as file:
            self.data = json.load(file)

    def get_context(self, topic):
        if topic == "organization":
            return json.dumps(self.data.get("organization", {}))
        return json.dumps(self.data.get(topic, {}))


def test_normalizer_does_not_rewrite_normal_english():
    repo = InfoRepository(REPO_PATH)

    text, matches = repo.normalize(
        "What is the experimental yak farm?"
    )

    assert text == "What is the experimental yak farm?"
    assert matches == []


def test_normalizer_preserves_full_canonical_phrases():
    repo = InfoRepository(REPO_PATH)

    for phrase in (
        "NRCY",
        "precision livestock management",
        "Nyukmadung",
        "Dr. Mihir Sarkar",
    ):
        text, matches = repo.normalize(phrase)
        assert text == phrase
        assert matches == []


def test_mispronounced_people_are_normalized():
    repo = InfoRepository(REPO_PATH)

    text, _ = repo.normalize("Who is Mehir Sakhar?")
    assert "Dr. Mihir Sarkar" in text

    text, _ = repo.normalize("Who is Moukhtar Hussain?")
    assert "Dr. Mokhtar Hussain" in text


def test_known_people_are_deterministic():
    async def run():
        repo = InfoRepository(REPO_PATH)
        curly = Curly(
            llm=ExplodingLLM(),
            knowledge=Knowledge(),
            info_repo=repo,
        )

        cases = {
            "Who is Mehir Sakhar?": "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak.",
            "What is Dr Mehir Sakhar's role?": "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak, Dirang.",
            "Who is Dr. Moukhtar Hussain?": "Dr. Mokhtar Hussain is Senior Scientist (AR&G) at ICAR–National Research Centre on Yak.",
            "Who is Dr. Vijay Paul?": "Dr. Vijay Paul is Principal Scientist (VPY) at ICAR–National Research Centre on Yak.",
            "Who is Dr. Dinamani Medhi?": "Dr. Dinamani Medhi is Principal Scientist (AN) at ICAR–National Research Centre on Yak.",
            "Who is Dr. Nupur Choudhury?": "Dr. Nupur Choudhury is Assistant Professor at Assam Don Bosco University.",
            "Who is Dr. Rupesh Mandal?": "Dr. Rupesh Mandal is an Assistant Professor at Assam Don Bosco University and is listed as the nodal officer and in-charge of the ICAR-NRCY Centre of Excellence at ADBU.",
            "Who is the Director of NRCY?": "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak, Dirang.",
            "Who is the nodal officer?": "Dr. Rupesh Mandal is an Assistant Professor at Assam Don Bosco University and is listed as the nodal officer and in-charge of the ICAR-NRCY Centre of Excellence at ADBU.",
            "Who is the senior scientist?": "Dr. Mokhtar Hussain is Senior Scientist (AR&G) at ICAR–National Research Centre on Yak.",
            "Who are the principal scientists?": "Dr. Vijay Paul and Dr. Dinamani Medhi are Principal Scientists at ICAR–National Research Centre on Yak.",
        }

        session_id = curly.create_session()
        for question, expected in cases.items():
            response = await curly.chat(
                session_id=session_id,
                text=question,
            )
            assert response.text == expected
            assert response.intent_source == "deterministic"

    asyncio.run(run())
