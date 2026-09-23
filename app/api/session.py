from fastapi import APIRouter

from app.core.curly import Curly
from app.models.schemas import SessionResponse


def create_session_router(
    curly: Curly
):

    router = APIRouter()

    @router.post(
        "/v1/session",
        response_model=SessionResponse
    )
    async def create_session():

        session_id = curly.create_session()

        return SessionResponse(
            session_id=session_id,
            active=True
        )

    @router.delete(
        "/v1/session/{session_id}"
    )

    async def delete_session(
     session_id: str
     ):
      
      curly.clear_session(
        session_id
      )

      return {
        "session_id": session_id,
        "active": False
      }

    @router.get(
        "/v1/session/{session_id}/state"
    )
    async def get_state(
        session_id: str
    ):

        state = curly.get_state(
            session_id
        )

        return {
            "session_id": session_id,
            "state": state
        }

    return router