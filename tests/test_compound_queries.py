import json
from pathlib import Path

import pytest

from app.core.curly import Curly


class ExplodingLLM:
    async def generate(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for deterministic compound institutional queries")

    async def generate_stream(self, *args, **kwargs):
        raise AssertionError("LLM must not be called for deterministic compound institutional queries")
        yield ""


class Knowledge:
    def __init__(self):
        path = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "data.json"
        with path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)


@pytest.mark.asyncio
async def test_icar_director_nodal_compound_query():
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    response = await curly.chat(
        session_id,
        "Can you explain what you mean by ICAR and can you talk about who is its director and nodal officer?",
    )

    assert response.intent_source == "deterministic"
    assert "ICAR stands for Indian Council of Agricultural Research." in response.text
    assert "Dr. Mihir Sarkar" in response.text
    assert "Dr. Rupesh Mandal" in response.text


@pytest.mark.asyncio
async def test_stream_compound_query_is_still_deterministic():
    curly = Curly(llm=ExplodingLLM(), knowledge=Knowledge())
    session_id = curly.create_session()

    chunks = []
    async for chunk in curly.stream_normal_response(
        session_id,
        "Can you explain what you mean by ICAR and can you talk about who is its director and nodal officer?",
    ):
        chunks.append(chunk)

    text = "".join(chunks)
    assert "ICAR stands for Indian Council of Agricultural Research." in text
    assert "Dr. Mihir Sarkar" in text
    assert "Dr. Rupesh Mandal" in text
