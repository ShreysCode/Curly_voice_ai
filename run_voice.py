import asyncio
from pathlib import Path

from app.config import settings
from app.core.curly import Curly
from app.knowledge.knowledge import KnowledgeBase
from app.llm.ollama import OllamaClient
from app.stt.service import STTService
from app.tts.service import TTSService
from app.voice.engine import CurlyVoiceEngine


async def main():
    base_dir = Path(__file__).resolve().parent

    knowledge = KnowledgeBase(
        base_dir / "app" / "knowledge" / "data.json"
    )

    llm = OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
    )

    print("Warming up Curly LLM...")
    try:
        await llm.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Reply with OK.",
                }
            ]
        )
        print("Curly LLM ready.")
    except Exception as error:
        print(f"WARNING: Curly LLM warm-up failed: {error}")
        print(f"Make sure Ollama is running at {settings.ollama_url}")
        print("Continuing startup; LLM requests will be retried when needed.")

    curly = Curly(
        llm=llm,
        knowledge=knowledge,
    )

    stt = STTService()
    tts = TTSService(output_dir="audio")

    engine = CurlyVoiceEngine(
        curly=curly,
        stt=stt,
        tts=tts,
    )

    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
