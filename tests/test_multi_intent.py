import json
import re
from pathlib import Path

import pytest

from app.core.curly import Curly


class ExplodingLLM:
    async def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for fully deterministic multi-question queries")

    async def generate_stream(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for fully deterministic multi-question queries")
        yield ""


class RecordingLLM:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def generate(self, messages, response_format=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)

        locked = re.findall(
            r"AUTHORITATIVE ANSWER:\s*(.+?)(?=\n\d+\. QUESTION:|\Z)",
            prompt,
            flags=re.DOTALL,
        )
        locked = [item.strip() for item in locked if item.strip()]

        if locked:
            return " ".join(locked) + " And the remaining question is answered from the supplied knowledge."
        return "The remaining questions are answered from the supplied knowledge."


class Knowledge:
    def __init__(self):
        path = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "data.json"
        with path.open("r", encoding="utf-8") as file:
            self.data = json.load(file)

    def get_context(self, topic):
        data = self.data.get(topic, {})
        return json.dumps(data, ensure_ascii=False)


@pytest.mark.parametrize(
    "query",
    [
        "What is ICAR and who is the Director?",
        "Who is the Director, who is the nodal officer, and where is NRCY located?",
        "What is ICAR, who is the Director, and who is the nodal officer?",
        "Can you explain what you mean by ICAR and can you talk about who is its director and nodal officer?",
        "Who is the Director and nodal officer?",
        "What are the lab hours and who is the lab in-charge?",
        "What time is it and who is the Director?",
        "What is today's date and who is the nodal officer?",
    ],
)
@pytest.mark.asyncio
async def test_fully_deterministic_multi_question_queries_do_not_call_llm(query):
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(session_id, query)

    assert response.intent_source == "deterministic"
    assert response.text


def test_compound_splitter_handles_general_patterns():
    curly = Curly.__new__(Curly)

    cases = {
        "What is ICAR and who is the Director?": 2,
        "Who is the Director and nodal officer?": 2,
        "Who is the Director, who is the nodal officer, and where is NRCY located?": 3,
        "What does NRCY research and why is it important?": 2,
        "Who is Dr. Vijay Paul and who is Dr. Nupur Choudhury?": 2,
        "What are the lab hours?": 1,
    }

    for query, expected_count in cases.items():
        parts = curly._split_compound_query(query)
        assert len(parts) == expected_count, (query, parts)


@pytest.mark.asyncio
async def test_multiple_named_people_in_one_question_are_all_answered():
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(
        session_id,
        "Tell me about Dr. Vijay Paul and Dr. Nupur Choudhury.",
    )

    assert response.intent_source == "deterministic"
    assert "Dr. Vijay Paul" in response.text
    assert "Dr. Nupur Choudhury" in response.text


@pytest.mark.asyncio
async def test_mixed_deterministic_and_llm_queries_use_one_llm_call():
    llm = RecordingLLM()
    curly = Curly(llm=llm, knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(
        session_id,
        "Who is the Director and explain what NRCY researches.",
    )

    assert llm.calls == 1
    assert "Dr. Mihir Sarkar" in response.text
    assert response.intent_source == "llm"


@pytest.mark.asyncio
async def test_all_unresolved_multi_question_query_uses_one_llm_call():
    llm = RecordingLLM()
    curly = Curly(llm=llm, knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(
        session_id,
        "What does NRCY research and why is it important?",
    )

    assert llm.calls == 1
    assert response.intent_source == "llm"
    assert response.text


@pytest.mark.asyncio
async def test_streaming_compound_deterministic_query_uses_same_router():
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    chunks = []
    async for chunk in curly.stream_normal_response(
        session_id,
        "Who is the Director and who is the nodal officer?",
    ):
        chunks.append(chunk)

    text = "".join(chunks)
    assert "Dr. Mihir Sarkar" in text
    assert "Dr. Rupesh Mandal" in text
