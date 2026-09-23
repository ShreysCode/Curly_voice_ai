from pathlib import Path
import time
import logging
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException
)

from fastapi.responses import FileResponse

from app.core.curly import Curly
from app.models.schemas import (
    CurlyState,
    TTSRequest
)

from app.tts.service import TTSService


def create_tts_router(
    tts: TTSService,
    curly: Curly
):

    router = APIRouter()

    @router.post("/v1/tts")
    async def synthesize(
        request: TTSRequest,
        background_tasks: BackgroundTasks
    ):

        try:

            curly.get_session(
                request.session_id
            )

            curly.set_state(
                request.session_id,
                CurlyState.SPEAKING
            )

            tts_start = time.perf_counter()

            audio_path = await tts.synthesize(
                request.text
            )

            tts_latency_ms = (
               time.perf_counter() - tts_start
            ) * 1000

            logging.getLogger("curly").info(
               "TTS generation: %.2f ms",
            tts_latency_ms,
            )    

            background_tasks.add_task(
                _delete_file,
                audio_path
            )

            return FileResponse(
                path=audio_path,
                media_type="audio/mpeg",
                filename=audio_path.name
            )

        except Exception as error:

            curly.set_state(
                request.session_id,
                CurlyState.ERROR
            )

            raise HTTPException(
                status_code=500,
                detail="TTS generation failed"
            ) from error

    return router


def _delete_file(
    path: Path
):

    try:

        path.unlink(
            missing_ok=True
        )

    except OSError:

        pass