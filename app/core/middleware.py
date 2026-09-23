import logging
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger("curly")


async def request_id_middleware(
    request: Request,
    call_next
):
    request_id = request.headers.get(
        "X-Request-ID"
    ) or str(uuid.uuid4())

    request.state.request_id = request_id

    try:

        response = await call_next(request)

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    except Exception:

        logger.exception(
            "Unhandled request error",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path
            }
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An internal server error occurred.",
                "request_id": request_id
            },
            headers={
                "X-Request-ID": request_id
            }
        )