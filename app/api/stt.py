import os
import tempfile

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)

from app.stt.service import STTService


def create_stt_router(
    stt: STTService
):

    router = APIRouter()

    @router.post(
        "/v1/stt"
    )
    async def transcribe(
        file: UploadFile = File(...)
    ):

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Audio file is required"
            )

        suffix = (
            os.path.splitext(
                file.filename
            )[1]
            or ".wav"
        )

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp:

                temp_path = temp.name

                content = await file.read()

                temp.write(content)

            result = stt.transcribe(
                temp_path
            )

            return result

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail="Speech recognition failed"
            ) from error

        finally:

            if temp_path and os.path.exists(
                temp_path
            ):

                os.remove(temp_path)

    return router