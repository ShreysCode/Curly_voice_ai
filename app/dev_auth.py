import asyncio

from app.config import settings
from app.core.curly import Curly
from app.knowledge.knowledge import KnowledgeBase
from app.llm.ollama import OllamaClient
from app.stt.service import STTService
from app.tts.service import TTSService
from app.voice.engine import CurlyVoiceEngine


async def main():

    from pathlib import Path

    base_dir = Path(__file__).resolve().parent

    knowledge = KnowledgeBase(
        base_dir
        / "app"
        / "knowledge"
        / "data.json"
    )

    llm = OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
    )

    curly = Curly(
        llm=llm,
        knowledge=knowledge,
    )

    stt = STTService()

    tts = TTSService(
        output_dir="audio"
    )

    engine = CurlyVoiceEngine(
        curly=curly,
        stt=stt,
        tts=tts,
    )

    print(
        f"Session: {engine.session_id}"
    )

    await engine.handle_auth_result(
        status="AUTHORIZED",
        name="Test User",
    )


if __name__ == "__main__":
    asyncio.run(main())