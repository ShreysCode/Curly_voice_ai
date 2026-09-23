import pytest

from app.core.curly import Curly
from app.models.schemas import (
    Command,
    CurlyState,
    ResponseType
)


class FakeLLM:

    async def generate(
        self,
        messages,
        response_format=None
    ):
        return """
        {
            "type": "command",
            "command": "FACE_AUTH",
            "text": "Sure! I'll verify you.",
            "confidence": 1.0
        }
        """

    async def health(self):
        return True


class FakeKnowledge:

    def get_context(
        self,
        topic=None
    ):
        return ""


@pytest.mark.asyncio
async def test_complete_authentication_flow():

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )

    session_id = curly.create_session()

    # -----------------------------------------
    # USER REQUEST
    # -----------------------------------------

    response = await curly.chat(
        session_id,
        "I need to get inside, can you check my identity?"
    )

    assert response.type == ResponseType.COMMAND
    assert response.command == Command.FACE_AUTH
    assert response.state == CurlyState.AUTHENTICATING

    # -----------------------------------------
    # SIMULATE AUTH BACKEND
    # -----------------------------------------

    auth_result = curly.handle_auth_result(
        session_id,
        {
            "status": "AUTHORIZED",
            "name": "Test User"
        }
    )

    assert auth_result.type == ResponseType.RESPONSE
    assert auth_result.command == Command.NONE
    assert auth_result.state == CurlyState.SPEAKING

    assert (
        "You're verified"
        in auth_result.text
    )

    assert (
        "Test User"
        in auth_result.text
    )

    @pytest.mark.parametrize(
    "status, expected",
    [
        (
            "UNAUTHORIZED",
            "not authorized"
        ),
        (
            "UNKNOWN_FACE",
            "couldn't identify"
        ),
        (
            "NO_FACE",
            "couldn't see a face"
        ),
        (
            "NETWORK_ERROR",
            "trouble reaching"
        ),
        (
            "TIMEOUT",
            "took too long"
        ),
        (
            "SERVER_ERROR",
            "currently unavailable"
        )
    ]
)
    
    def test_authentication_errors(
    status,
    expected
):

     curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )

     session_id = curly.create_session()

     result = curly.handle_auth_result(
        session_id,
        {
            "status": status
        }
    )

     assert expected in result.text.lower()
     assert result.state == CurlyState.SPEAKING