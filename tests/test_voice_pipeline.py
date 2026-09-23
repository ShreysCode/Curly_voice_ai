import pytest

from app.core.curly import Curly
from app.models.schemas import (
    Command,
    ResponseType
)


class FakeSTT:

    def transcribe(
        self,
        audio_path
    ):

        return {
            "text": "Can you verify me?",
            "language": "en",
            "language_probability": 0.99
        }


class FakeLLM:

    async def generate(
        self,
        messages,
        response_format=None
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
async def test_voice_to_face_auth():

    stt = FakeSTT()

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )

    transcript = stt.transcribe(
        "fake-audio.wav"
    )

    assert (
        transcript["text"]
        == "Can you verify me?"
    )

    response = await curly.chat(
        session_id=curly.create_session(),
        text=transcript["text"]
    )

    assert response.type == ResponseType.COMMAND
    assert response.command == Command.FACE_AUTH