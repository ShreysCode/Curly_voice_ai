import pytest

from app.core.curly import Curly
from app.models.schemas import (
    Command,
    CurlyState,
    ResponseType,
)


class FakeSTT:

    def transcribe(self, audio_path):

        return {
            "text": "Can you verify me?",
            "language": "en",
            "language_probability": 0.99,
        }


class FakeLLM:

    async def generate(
        self,
        messages,
        response_format=None,
    ):

        return """
        {
            "type": "command",
            "command": "FACE_AUTH",
            "text": "Sure! I'll verify you.",
            "confidence": 1.0
        }
        """


class FakeKnowledge:

    def get_context(self, topic=None):
        return ""


@pytest.mark.asyncio
async def test_voice_authentication_flow():

    stt = FakeSTT()

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge(),
    )

    session_id = curly.create_session()

    # -----------------------------
    # AUDIO -> STT
    # -----------------------------

    transcription = stt.transcribe(
        "fake-audio.m4a"
    )

    assert (
        transcription["text"]
        == "Can you verify me?"
    )

    # -----------------------------
    # STT -> CURLY
    # -----------------------------

    response = await curly.chat(
        session_id=session_id,
        text=transcription["text"],
    )

    assert response.type == ResponseType.COMMAND
    assert response.command == Command.FACE_AUTH
    assert response.state == CurlyState.AUTHENTICATING

    # -----------------------------
    # SECURITY BACKEND RESULT
    # -----------------------------

    auth_response = curly.handle_auth_result(
        session_id=session_id,
        data={
            "status": "AUTHORIZED",
            "name": "Test User",
        },
    )

    assert auth_response.type == ResponseType.RESPONSE
    assert auth_response.command == Command.NONE
    assert auth_response.state == CurlyState.SPEAKING

    assert (
        "You're verified"
        in auth_response.text
    )

    assert (
        "Test User"
        in auth_response.text
    )