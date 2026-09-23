from app.core.intent import detect_command
from app.models.schemas import Command


def test_face_auth():

    result = detect_command(
        "Verify me"
    )

    assert result is not None
    assert result.command == Command.FACE_AUTH


def test_face_auth_scan():

    result = detect_command(
        "Scan my face"
    )

    assert result is not None
    assert result.command == Command.FACE_AUTH


def test_time():

    result = detect_command(
        "What time is it?"
    )

    assert result is not None
    assert result.command == Command.GET_TIME


def test_weather():

    result = detect_command(
        "What's the weather?"
    )

    assert result is not None
    assert result.command == Command.GET_WEATHER


def test_goodbye():

    result = detect_command(
        "Goodbye Curly"
    )

    assert result is not None
    assert result.command == Command.END_CONVERSATION


def test_normal_conversation():

    result = detect_command(
        "How are you?"
    )

    assert result is None

def test_face_auth_variations():

    phrases = [
        "Can you check my identity?",
        "I need to get inside, can you scan me?",
        "Could you authenticate me?",
        "Can I enter?",
        "Please recognize me",
        "I need access, check me"
    ]

    for text in phrases:

        result = detect_command(text)

        assert result is not None
        assert result.command == Command.FACE_AUTH

def test_lab_information():

    phrases = [
        "When does the lab open?",
        "When does the lab close?",
        "Where is the lab?",
        "Who is the lab in-charge?",
        "What time does the laboratory close?"
    ]

    for text in phrases:

        result = detect_command(text)

        assert result is not None
        assert result.command in {
            Command.GET_LAB_INFO,
            Command.GET_LAB_IN_CHARGE,
            Command.GET_LAB_NODAL_OFFICER,
            Command.GET_LAB_HOURS,
            Command.GET_LAB_LOCATION,
            Command.GET_LAB_CONTACT,
            Command.GET_LAB_EMAIL,
        }


def test_lab_mention_is_not_lab_command():

    phrases = [
        "How are things at the lab?",
        "I'm working in the lab today.",
        "Is everyone at the lab?"
    ]

    for text in phrases:

        result = detect_command(text)

        assert result is None


def test_lab_in_charge():
    result = detect_command(
        "Who is the in-charge of the lab?"
    )
    assert result.command == Command.GET_LAB_IN_CHARGE


def test_lab_nodal_officer():
    result = detect_command(
        "Who is the nodal officer?"
    )
    assert result.command == Command.GET_LAB_NODAL_OFFICER


def test_lab_hours():
    result = detect_command(
        "When does the lab open?"
    )
    assert result.command == Command.GET_LAB_HOURS


def test_time():
    result = detect_command(
        "What time is it right now?"
    )
    assert result.command == Command.GET_TIME


def test_date():
    result = detect_command(
        "What's today's date?"
    )
    assert result.command == Command.GET_DATE

def test_nodal_officer_variations():
    phrases = [
        "Who is the nodal officer?",
        "Who is the nodal officer of the lab?",
        "Who is the nodal officer for the lab?",
        "Who is our nodal officer?",
        "Who handles this as the nodal officer?",
        "Who's the nodal officer?",
    ]

    for text in phrases:
        result = detect_command(text)

        assert result is not None
        assert result.command == Command.GET_LAB_NODAL_OFFICER


def test_in_charge_variations():
    phrases = [
        "Who is in charge?",
        "Who is the lab in charge?",
        "Who is in-charge of the lab?",
        "Who is the in-charge of this lab?",
        "Who runs the lab?",
        "Who is the head of the lab?",
    ]

    for text in phrases:
        result = detect_command(text)

        assert result is not None
        assert result.command == Command.GET_LAB_IN_CHARGE


def test_lab_hours_variations():
    phrases = [
        "When does the lab open?",
        "When does lab open?",
        "What time does the laboratory close?",
        "What are the lab timings?",
        "What are the working hours of the laboratory?",
    ]

    for text in phrases:
        result = detect_command(text)

        assert result is not None
        assert result.command == Command.GET_LAB_HOURS