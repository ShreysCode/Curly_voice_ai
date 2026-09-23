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
        return "Hello!"

    async def health(self):
        return True


class FakeKnowledge:

    def get_context(self):
        return ""


@pytest.fixture
def curly():

    return Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge()
    )


def test_authorized(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "AUTHORIZED",
            "name": "Test User"
        }
    )

    assert result.type == ResponseType.RESPONSE

    assert (
        result.text
        == "You're verified. Welcome, Test User!"
    )

    assert result.command == Command.NONE

    assert result.state == CurlyState.SPEAKING


def test_unauthorized(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "UNAUTHORIZED"
        }
    )

    assert (
        result.text
        == "I'm sorry, you're not authorized to enter."
    )

    assert result.state == CurlyState.SPEAKING


def test_unknown_face(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "UNKNOWN_FACE"
        }
    )

    assert (
        result.text
        == "I couldn't identify you. Please try again."
    )


def test_no_face(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "NO_FACE"
        }
    )

    assert (
        result.text
        == "I couldn't see a face. Please try again."
    )


def test_network_error(curly):

    session_id = curly.create_session()

    result = curly.handle_auth_result(
        session_id,
        {
            "status": "NETWORK_ERROR"
        }
    )

    assert (
        result.text
        == "I'm having trouble reaching the verification service."
    )