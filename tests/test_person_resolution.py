import json
from pathlib import Path

import pytest

from app.core.curly import Curly


class ExplodingLLM:
    async def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for known person/role queries")

    async def generate_stream(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for known person/role queries")
        yield ""


class Knowledge:
    def __init__(self):
        path = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "data.json"
        with path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who is the Director of NRCY?", "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak, Dirang."),
        ("Who is Dr. Mihir Sarkar?", "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak."),
        ("What is Dr. Mihir Sarkar's role?", "Dr. Mihir Sarkar is the Director of ICAR–National Research Centre on Yak, Dirang."),
        ("Who is the nodal officer?", "Dr. Rupesh Mandal is an Assistant Professor at Assam Don Bosco University and is listed as the nodal officer and in-charge of the ICAR-NRCY Centre of Excellence at ADBU."),
        ("Who is Dr. Rupesh Mandal?", "Dr. Rupesh Mandal is an Assistant Professor at Assam Don Bosco University and is listed as the nodal officer and in-charge of the ICAR-NRCY Centre of Excellence at ADBU."),
        ("Who is Dr. Vijay Paul?", "Dr. Vijay Paul is Principal Scientist (VPY) at ICAR–National Research Centre on Yak."),
        ("Who is Dr. Nupur Choudhury?", "Dr. Nupur Choudhury is Assistant Professor at Assam Don Bosco University."),
        (
        "Who is the senior scientist?",
        "Dr. Mokhtar Hussain is Senior Scientist (AR&G) at ICAR–National Research Centre on Yak.",
    ),
    (
        "Do you have information about Dr. Vijay Paul?",
        "Dr. Vijay Paul is Principal Scientist (VPY) at ICAR–National Research Centre on Yak.",
    ),
    (
        "Who is Dr. Moukhtar Hussain?",
        "Dr. Mokhtar Hussain is Senior Scientist (AR&G) at ICAR–National Research Centre on Yak.",
    ),
    (
        "Tell me about Dr. Vijay Paul.",
        "Dr. Vijay Paul is Principal Scientist (VPY) at ICAR–National Research Centre on Yak.",
    ),
    (
        "Who is the senior scientist?",
        "Dr. Mokhtar Hussain is Senior Scientist (AR&G) at ICAR–National Research Centre on Yak.",
    ),
    (
        "Who is the principal scientist?",
        "Dr. Vijay Paul and Dr. Dinamani Medhi are Principal Scientists at ICAR–National Research Centre on Yak.",
    ),
    ],
)
async def test_known_person_queries_are_deterministic(question, expected):
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(session_id=session_id, text=question)

    assert response.text == expected
    assert response.intent_source == "deterministic"
