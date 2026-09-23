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
        return (
            "I'm Curly! I'm here to chat, "
            "answer questions, and help you."
        )

    async def health(self):
        return True


class FakeKnowledge:

    def get_context(self):
        return """
        Organization:
        ICAR

        Laboratory:
        Opening time: 09:00
        Closing time: 17:00
        Location: Main Laboratory
        """


@pytest.fixture
def curly():

    return Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )


@pytest.mark.asyncio
async def test_normal_conversation(curly):

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "How are you?"
    )

    assert result.type == ResponseType.RESPONSE
    assert result.command == Command.NONE
    assert result.state == CurlyState.SPEAKING
    assert result.text


@pytest.mark.asyncio
async def test_face_auth_command(curly):

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "Can you verify me?"
    )

    assert result.type == ResponseType.COMMAND
    assert result.command == Command.FACE_AUTH
    assert result.state == CurlyState.AUTHENTICATING


@pytest.mark.asyncio
async def test_time_command(curly):

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "What time is it?"
    )

    assert result.command == Command.GET_TIME


@pytest.mark.asyncio
async def test_weather_command(curly):

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "What's the weather?"
    )

    assert result.command == Command.GET_WEATHER


@pytest.mark.asyncio
async def test_end_conversation(curly):

    session_id = curly.create_session()

    result = await curly.chat(
        session_id,
        "Goodbye Curly"
    )

    assert result.command == Command.END_CONVERSATION
    assert result.state == CurlyState.IDLE


def test_authorization_never_inferred(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "UNAUTHORIZED"
        }
    )

    assert "authorized" in result.text.lower()