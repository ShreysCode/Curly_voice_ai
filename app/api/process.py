import json
import os
import tempfile
import time

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.core.curly import Curly
from app.models.schemas import CurlyState
from app.stt.service import STTService


MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB


def create_process_router(
    curly: Curly,
    stt: STTService,
):

    router = APIRouter()

    @router.post("/v1/process")
    async def process_audio(
        session_id: str = Form(...),
        file: UploadFile = File(...),
        context: str | None = Form(None),
    ):

        # --------------------------------
        # SESSION VALIDATION
        # --------------------------------

        if session_id not in curly.sessions:

            raise HTTPException(
                status_code=404,
                detail={
                    "code": "INVALID_SESSION",
                    "message": "Session not found.",
                },
            )

        # --------------------------------
        # FILE VALIDATION
        # --------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_AUDIO",
                    "message": "Audio file is required.",
                },
            )

        # --------------------------------
        # CONTEXT
        # --------------------------------

        environment_context = {}

        if context:

            try:

                environment_context = json.loads(
                    context
                )

                if not isinstance(
                    environment_context,
                    dict,
                ):
                    raise ValueError

            except (
                json.JSONDecodeError,
                ValueError,
            ):

                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "INVALID_CONTEXT",
                        "message": "Context must be valid JSON.",
                    },
                )

        # --------------------------------
        # TEMP FILE
        # --------------------------------

        suffix = (
            os.path.splitext(
                file.filename
            )[1]
            or ".wav"
        )

        temp_path = None

        try:

            # --------------------------------
            # PROCESSING STATE
            # --------------------------------

            curly.set_state(
                session_id,
                CurlyState.PROCESSING,
            )

            # --------------------------------
            # SAVE AUDIO
            # --------------------------------

            content = await file.read()

            if not content:

                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "EMPTY_AUDIO",
                        "message": "Audio file is empty.",
                    },
                )

            if len(content) > MAX_AUDIO_BYTES:

                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "AUDIO_TOO_LARGE",
                        "message": (
                            "Audio file exceeds the "
                            "10 MB limit."
                        ),
                    },
                )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:

                temp_path = temp.name
                temp.write(content)

            # --------------------------------
            # STT
            # --------------------------------

            stt_start = time.perf_counter()

            try:

                transcription = stt.transcribe(
                    temp_path
                )

            except ValueError as error:

               curly.set_state(
        session_id,
        CurlyState.LISTENING,
    )

               raise HTTPException(
        status_code=422,
        detail={
            "code": "ENGLISH_ONLY",
            "message": "Please speak in English.",
        },
    ) from error

            except Exception as error:

                curly.set_state(
                    session_id,
                    CurlyState.ERROR,
                )

                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "STT_FAILED",
                        "message": (
                            "Speech recognition failed."
                        ),
                    },
                ) from error

            stt_latency_ms = (
                time.perf_counter()
                - stt_start
            ) * 1000

            text = transcription.get(
                "text",
                "",
            ).strip()

            if not text:

                curly.set_state(
                    session_id,
                    CurlyState.LISTENING,
                )

                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "NO_SPEECH",
                        "message": "No speech was detected.",
                    },
                )

            # --------------------------------
            # CURLY
            # --------------------------------

            curly_start = time.perf_counter()

            try:

                curly_response = await curly.chat(
                    session_id=session_id,
                    text=text,
                    context=environment_context,
                )

            except Exception as error:

                curly.set_state(
                    session_id,
                    CurlyState.ERROR,
                )

                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "CURLY_PROCESSING_FAILED",
                        "message": (
                            "Curly could not process "
                            "the request."
                        ),
                    },
                ) from error

            curly_latency_ms = (
                time.perf_counter()
                - curly_start
            ) * 1000

            # --------------------------------
            # SUCCESS
            # --------------------------------

            return {
                "session_id": session_id,
                "transcript": {
                    "text": text,
                    "language": transcription.get(
                        "language",
                        "",
                    ),
                    "language_probability": (
                        transcription.get(
                            "language_probability",
                            0.0,
                        )
                    ),
                },
                "response": (
                    curly_response.model_dump()
                ),
                "latency_ms": {
                    "stt": round(
                        stt_latency_ms,
                        2,
                    ),
                    "curly": round(
                        curly_latency_ms,
                        2,
                    ),
                    "total": round(
                        stt_latency_ms
                        + curly_latency_ms,
                        2,
                    ),
                },
            }

        except HTTPException:
            raise

        except Exception as error:

            curly.set_state(
                session_id,
                CurlyState.ERROR,
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "code": "VOICE_PROCESSING_FAILED",
                    "message": (
                        "Voice processing failed."
                    ),
                },
            ) from error

        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:

                    os.remove(temp_path)

                except OSError:
                    pass

    return router