import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.core.curly import Curly, CurlyState
from app.stt.service import STTService
from app.tts.service import TTSService


MAX_AUDIO_BYTES = 10 * 1024 * 1024


def create_voice_router(
    curly: Curly,
    stt: STTService,
    tts: TTSService,
):
    router = APIRouter()

    @router.post("/v1/voice")
    async def voice(
        session_id: str = Form(...),
        file: UploadFile = File(...),
        context: str | None = Form(None),
    ):
        if session_id not in curly.sessions:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "INVALID_SESSION",
                    "message": "Session not found.",
                },
            )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_AUDIO",
                    "message": "Audio file is required.",
                },
            )

        temp_path = None

        try:
            environment_context = {}

            if context:
                import json

                try:
                    environment_context = json.loads(context)
                except json.JSONDecodeError as error:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "code": "INVALID_CONTEXT",
                            "message": "Context must be valid JSON.",
                        },
                    ) from error

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
                        "message": "Audio file exceeds the 10 MB limit.",
                    },
                )

            suffix = (
                os.path.splitext(file.filename)[1]
                or ".wav"
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:
                temp_path = temp.name
                temp.write(content)

            # -----------------------------
            # STT
            # -----------------------------

            try:
                transcription = stt.transcribe(
                    temp_path
                )
            except Exception as error:
                curly.set_state(
                    session_id,
                    CurlyState.ERROR,
                )

                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "STT_FAILED",
                        "message": "Speech recognition failed.",
                    },
                ) from error

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

            # -----------------------------
            # CURLY
            # -----------------------------

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
                        "message": "Curly could not process the request.",
                    },
                ) from error

            # -----------------------------
            # TTS
            # -----------------------------

            # Do not synthesize an empty response.
            if not curly_response.text:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "EMPTY_RESPONSE",
                        "message": "Curly produced no spoken response.",
                    },
                )

            curly.set_state(
                session_id,
                CurlyState.SPEAKING,
            )

            try:
                audio_path = await tts.synthesize(
                    curly_response.text
                )
            except Exception as error:
                curly.set_state(
                    session_id,
                    CurlyState.ERROR,
                )

                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "TTS_FAILED",
                        "message": "Speech generation failed.",
                    },
                ) from error

            # -----------------------------
            # RETURN AUDIO
            # -----------------------------

            response = FileResponse(
                path=audio_path,
                media_type="audio/mpeg",
                filename=audio_path.name,
                headers={
                    "X-Curly-Session-Id": session_id,
                    "X-Curly-Transcript": text,
                    "X-Curly-Type": (
                        curly_response.type.value
                    ),
                    "X-Curly-Command": (
                        curly_response.command.name
                    ),
                    "X-Curly-State": (
                        curly_response.state.value
                    ),
                },
            )

            return response

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
                    "message": "Voice processing failed.",
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