import pytest

from app.core.curly import Curly
from app.models.schemas import (
    Command,
    CurlyState,
    ResponseType
)


class FakeLLM:

    async def generate(
        self,
        messages,
        response_format=None
    ):

        prompt = messages[-1]["content"]

        if "opening_time" in prompt:

            return "The lab opens at 9 AM."

        return "I don't have that information yet."


class FakeKnowledge:

    def get_context(
        self,
        topic=None
    ):

        if topic == "lab":

            return """
            {
                "lab": {
                    "opening_time": "09:00",
                    "closing_time": "17:00",
                    "location": "Main Laboratory"
                }
            }
            """

        return ""


@pytest.mark.asyncio
async def test_lab_opening_question():  

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "When does the lab open?"
    )

    assert result.type == ResponseType.RESPONSE
    assert result.command == Command.NONE
    assert result.state == CurlyState.SPEAKING
    assert "9 AM" in result.text