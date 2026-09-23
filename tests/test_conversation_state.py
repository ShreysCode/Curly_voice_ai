import pytest

from app.core.curly import Curly
from app.models.schemas import (
    CurlyState,
    ResponseType,
)


class FakeLLM:

    async def generate(
        self,
        messages,
        response_format=None,
    ):
        return "Hello! How can I help?"


class FakeKnowledge:

    def get_context(self, topic=None):
        return ""


@pytest.mark.asyncio
async def test_conversation_state_flow():

    curly = Curly(
        llm=FakeLLM(),
        knowledge=FakeKnowledge(),
    )

    session_id = curly.create_session()

    # -----------------------------
    # INITIAL STATE
    # -----------------------------

    session = curly.get_session(session_id)

    assert session.state == CurlyState.IDLE

    # -----------------------------
    # WAKE WORD
    # -----------------------------

    result = await curly.handle_event(
        session_id=session_id,
        event="WAKE_WORD",
        data={},
    )

    assert result.state == CurlyState.AWAKE

    # -----------------------------
    # LISTENING STARTED
    # -----------------------------

    result = await curly.handle_event(
        session_id=session_id,
        event="LISTENING_STARTED",
        data={},
    )

    assert result.state == CurlyState.LISTENING

    # -----------------------------
    # LISTENING STOPPED
    # -----------------------------

    result = await curly.handle_event(
        session_id=session_id,
        event="LISTENING_STOPPED",
        data={},
    )

    assert result.state == CurlyState.PROCESSING

    # -----------------------------
    # ACTUAL CONVERSATION
    # -----------------------------

    result = await curly.chat(
        session_id=session_id,
        text="How are you?",
    )

    assert result.type == ResponseType.RESPONSE
    assert result.state == CurlyState.SPEAKING

    # -----------------------------
    # TIMEOUT
    # -----------------------------

    result = await curly.handle_event(
        session_id=session_id,
        event="TIMEOUT",
        data={},
    )

    assert result.state == CurlyState.IDLE
    