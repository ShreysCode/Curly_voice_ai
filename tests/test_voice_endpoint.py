import pytest
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.voice import create_voice_router
from app.core.curly import Curly


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


class FakeTTS:

    async def synthesize(self, text):

        path = Path("tests/fake-response.mp3")
        path.write_bytes(b"fake-audio")

        return path


@pytest.mark.asyncio
async def test_voice_endpoint():

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge(),
    )

    stt = FakeSTT()
    tts = FakeTTS()

    session_id = curly.create_session()

    app = FastAPI()

    app.include_router(
        create_voice_router(
            curly=curly,
            stt=stt,
            tts=tts,
        )
    )

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.post(
            "/v1/voice",
            data={
                "session_id": session_id,
            },
            files={
                "file": (
                    "test.wav",
                    b"fake-audio",
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200
    assert response.headers[
        "X-Curly-Command"
    ] == "FACE_AUTH"

    assert response.headers[
        "X-Curly-State"
    ] == "AUTHENTICATING"

    fake_file = Path(
        "tests/fake-response.mp3"
    )

    fake_file.unlink(
        missing_ok=True
    )