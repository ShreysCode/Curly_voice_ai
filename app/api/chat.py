from fastapi import APIRouter, HTTPException

from app.core.curly import Curly
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EventRequest,
    EventResponse
)


def create_chat_router(
    curly: Curly
):

    router = APIRouter()

    # -----------------------------------------
    # CHAT
    # -----------------------------------------

    @router.post(
        "/v1/chat",
        response_model=ChatResponse
    )
    async def chat(
        request: ChatRequest
    ):

        try:

            result = await curly.chat(
                session_id=request.session_id,
                text=request.text,
                context=request.context
            )

            return ChatResponse(
                session_id=request.session_id,
                **result.model_dump()
            )

        except Exception as error:
            
            import logging

            logging.exception(
                "Curly processing failed"
            )

            raise HTTPException(
                status_code=500,
                detail=str(error)
            ) from error

    # -----------------------------------------
    # ANDROID EVENTS
    # -----------------------------------------

    @router.post(
        "/v1/events",
        response_model=EventResponse
    )
    async def event(
        request: EventRequest
    ):

        try:

            result = await curly.handle_event(
                session_id=request.session_id,
                event=request.event.value,
                data=request.data
            )

            return EventResponse(
                session_id=request.session_id,
                type=result.type,
                command=result.command,
                text=result.text
            )

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail="Curly event processing failed"
            ) from error

    # -----------------------------------------
    # RETURN ROUTER
    # -----------------------------------------

    return router