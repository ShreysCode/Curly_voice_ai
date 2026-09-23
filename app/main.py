from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat import create_chat_router
from app.api.process import create_process_router
from app.api.session import create_session_router
from app.api.stt import create_stt_router
from app.api.tts import create_tts_router

from app.config import settings
from app.core.curly import Curly

from app.knowledge.knowledge import KnowledgeBase
from app.llm.ollama import OllamaClient

from app.stt.service import STTService
from app.tts.service import TTSService

from app.core.logging import configure_logging
from app.core.middleware import request_id_middleware

from app.api.voice import create_voice_router

configure_logging()


BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize heavyweight Curly services when the application starts,
    instead of during module import.
    """

    # -----------------------------------------
    # KNOWLEDGE
    # -----------------------------------------

    knowledge = KnowledgeBase(
        BASE_DIR / "knowledge" / "data.json"
    )

    # -----------------------------------------
    # LLM
    # -----------------------------------------

    llm = OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model
    )

    # -----------------------------------------
    # CURLY
    # -----------------------------------------

    curly = Curly(
        llm=llm,
        knowledge=knowledge
    )

    # -----------------------------------------
    # TTS
    # -----------------------------------------

    tts = TTSService(
        output_dir="audio"
    )

    # -----------------------------------------
    # STT
    # -----------------------------------------

    stt = STTService()

    # -----------------------------------------
    # Store services on app.state
    # -----------------------------------------

    app.state.knowledge = knowledge
    app.state.llm = llm
    app.state.curly = curly
    app.state.tts = tts
    app.state.stt = stt

    # -----------------------------------------
    # Register routers
    # -----------------------------------------

    app.include_router(
        create_chat_router(curly)
    )

    app.include_router(
        create_tts_router(
            tts=tts,
            curly=curly
        )
    )

    app.include_router(
        create_stt_router(stt)
    )

    app.include_router(
        create_process_router(
            curly=curly,
            stt=stt
        )
    )

    app.include_router(
    create_voice_router(
        curly=curly,
        stt=stt,
        tts=tts,
    )
)

    app.include_router(
        create_session_router(curly)
    )


    yield

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    app.state.knowledge = None
    app.state.llm = None
    app.state.curly = None
    app.state.tts = None
    app.state.stt = None

    await llm.close()

app = FastAPI(
    title="Curly AI Backend",
    version="1.2.0",
    lifespan=lifespan
)

app.middleware("http")(
    request_id_middleware
)


@app.get("/")
async def root():

    return {
        "service": "curly-ai",
        "status": "running",
        "version": "1.2.0"
    }


@app.get("/health")
async def health():

    llm = getattr(
        app.state,
        "llm",
        None
    )

    if llm is None:

        return {
            "service": "curly-ai",
            "status": "starting",
            "ollama": False,
            "model": settings.ollama_model,
            "stt": False,
            "tts": False
        }

    ollama_status = await llm.health()

    return {
        "service": "curly-ai",

        "status": (
            "healthy"
            if ollama_status
            else "degraded"
        ),

        "ollama": ollama_status,

        "model": settings.ollama_model,

        "stt": (
            app.state.stt is not None
        ),

        "tts": (
            app.state.tts is not None
        )
    }