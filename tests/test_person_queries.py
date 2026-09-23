import pytest

from app.core.curly import Curly
from app.models.schemas import Command, ResponseType


class FakeLLM:
    async def generate(self, messages, response_format=None):
        raise AssertionError(
            "LLM should not be called for known person queries"
        )


class FakeKnowledge:
    data = {
        "lab": {
            "nodal_officer": "Dr. Rupesh Mandal",
            "in_charge": "Dr. Rupesh Mandal",
            "opening_time": "9 AM",
            "closing_time": "5 PM",
            "working_days": "Monday to Saturday",
            "location": "ADBU",
            "contact": "Dr. Rupesh Mandal",
            "email": "xxx@uni.ac.in",
        }
    }


@pytest.mark.asyncio
async def test_rupesh_mandal():
    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge(),
    )

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "Who is Dr. Rupesh Mandal?",
    )

    assert result.type == ResponseType.RESPONSE
    assert "Dr. Rupesh Mandal" in result.text
    assert "Assistant Professor" in result.text


@pytest.mark.asyncio
async def test_mihir_sarkar():
    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge(),
    )

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "Who is Dr. Mehir Sakhar?",
    )

    assert result.type == ResponseType.RESPONSE
    assert "Dr. Mihir Sarkar" in result.text
    assert "Director" in result.text