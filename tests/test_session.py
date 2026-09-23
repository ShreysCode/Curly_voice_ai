from app.core.session import Session
from app.models.schemas import CurlyState


def test_session_initial_state():

    session = Session(
        session_id="test-session"
    )

    assert session.state == CurlyState.IDLE


def test_session_state_change():

    session = Session(
        session_id="test-session"
    )

    session.set_state(
        CurlyState.PROCESSING
    )

    assert session.state == CurlyState.PROCESSING


def test_session_history():

    session = Session(
        session_id="test-session"
    )

    session.history.append({
        "role": "user",
        "content": "Hello"
    })

    assert len(session.history) == 1
    assert session.history[0]["content"] == "Hello"