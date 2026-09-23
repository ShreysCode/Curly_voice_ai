import re

from app.models.schemas import Command, LLMDecision


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize(text: str) -> str:
    """
    Normalize speech/STT text into a consistent form.
    """

    text = text.lower().strip()

    # Expand common contractions first.
    contractions = {
        "what's": "what is",
        "who's": "who is",
        "how's": "how is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "that's": "that is",
        "it's": "it is",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
    }

    for contraction, expansion in contractions.items():
        text = text.replace(
            contraction,
            expansion,
        )

    # Possessives:
    # today's   -> todays
    # laboratory's -> laboratorys
    text = re.sub(
        r"\b([a-z0-9]+)'s\b",
        r"\1s",
        text,
    )

    # Normalize hyphens.
    text = text.replace("-", " ")

    # Remove remaining punctuation.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Collapse whitespace.
    return " ".join(text.split())


def _decision(
    command: Command,
    text: str = "",
    confidence: float = 1.0,
) -> LLMDecision:
    """
    Create one consistent deterministic intent result.
    """

    return LLMDecision(
        type="command",
        command=command,
        text=text,
        confidence=confidence,
    )


def _contains_any(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    return any(
        phrase in text
        for phrase in phrases
    )


# =========================================================
# MAIN INTENT DETECTOR
# =========================================================

def detect_command(text: str) -> LLMDecision | None:

    normalized = _normalize(text)

    if not normalized:
        return None

    # =====================================================
    # 1. FACE AUTHENTICATION
    # =====================================================

    if _contains_any(
        normalized,
        (
            "verify me",
            "authenticate me",
            "check my identity",
            "check my id",
            "scan my face",
            "scan me",
            "check me",
            "recognize me",
            "recognise me",
            "let me in",
            "can i enter",
            "can i get in",
            "open the door",
            "check if i can enter",
            "check whether i can enter",
            "check if im allowed in",
            "check if i am allowed in",
            "i need access",
            "i need to get inside",
            "i need to enter",
            "verify my access",
            "verify access",
            "authenticate my access",
        ),
    ):
        return _decision(
            Command.FACE_AUTH,
            "Sure. I'll verify you.",
        )

    # =====================================================
    # 2. TIME
    # =====================================================

    time_phrases = (
        "what time is it",
        "what is the time",
        "whats the time",
        "tell me the current time",
        "current time",
        "what time is it right now",
        "time right now",
        "time now",
        "tell me what time it is",
        "tell me the time",
        "can you tell me the time",
        "could you tell me the time",
    )

    if _contains_any(normalized, time_phrases):
        return _decision(
            Command.GET_TIME,
            "",
        )

    # =====================================================
    # 3. DATE
    # =====================================================

    date_phrases = (
        "what is todays date",
        "whats todays date",
        "what is the date today",
        "whats the date today",
        "todays date",
        "todays date",
        "current date",
        "what date is it",
        "what is the date",
        "tell me todays date",
        "tell me the date",
        "can you tell me todays date",
        "can you tell me the date",
    )

    if _contains_any(normalized, date_phrases):
        return _decision(
            Command.GET_DATE,
            "",
        )

    # =====================================================
    # 4. WEATHER
    # =====================================================

    if _contains_any(
        normalized,
        (
            "weather",
            "temperature outside",
            "what is the temperature",
            "whats the temperature",
            "what is the weather",
            "whats the weather",
            "how is the weather",
            "hows the weather",
            "current weather",
            "weather right now",
            "tell me the weather",
            "weather today",
            "temperature right now",
            "temperature today",
        ),
    ):
        return _decision(
            Command.GET_WEATHER,
            "",
        )

    # =====================================================
    # 5. END CONVERSATION
    # =====================================================

    if _contains_any(
        normalized,
        (
            "goodbye",
            "good bye",
            "bye",
            "see you",
            "see you later",
            "stop",
            "exit",
            "quit",
            "end conversation",
            "thats all",
            "i m done",
            "im done",
            "i am done",
            "stop talking",
            "you can stop",
        ),
    ):
        return _decision(
            Command.END_CONVERSATION,
            "Goodbye!",
        )

    # =====================================================
    # 6. LAB — NODAL OFFICER
    # =====================================================
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT require:
    #     "of the lab"
    #
    # These all become the same intent:
    #
    #   Who is the nodal officer?
    #   Who is the nodal officer of the lab?
    #   Who is the lab's nodal officer?
    #   Who's nodal officer?
    #
    # This is the key improvement you asked for.
    # =====================================================

    if _contains_any(
        normalized,
        (
            "nodal officer",
            "nodalofficer",
            "nodal person",
            "nodal responsible person",
        ),
    ):
        return _decision(
            Command.GET_LAB_NODAL_OFFICER,
            "",
        )

    # =====================================================
    # 7. LAB — IN CHARGE
    # =====================================================

    in_charge_match = (
        "in charge" in normalized
        or "incharge" in normalized
        or "person in charge" in normalized
        or "person responsible for the lab" in normalized
        or "head of the lab" in normalized
        or "lab head" in normalized
        or "head of laboratory" in normalized
        or "laboratory head" in normalized
        or "who runs the lab" in normalized
        or "who runs the laboratory" in normalized
    )

    if in_charge_match:
        return _decision(
            Command.GET_LAB_IN_CHARGE,
            "",
        )

    # =====================================================
    # 8. LAB — HOURS
    # =====================================================

    lab_hours_match = (

        # Open / opening
        "when does the lab open" in normalized
        or "when does lab open" in normalized
        or "when does the laboratory open" in normalized
        or "when does laboratory open" in normalized
        or "what time does the lab open" in normalized
        or "what time does lab open" in normalized
        or "what time does the laboratory open" in normalized
        or "what time does laboratory open" in normalized
        or "when is the lab open" in normalized
        or "when is laboratory open" in normalized
        or "when is the laboratory open" in normalized

        # Close / closing
        or "when does the lab close" in normalized
        or "when does lab close" in normalized
        or "when does the laboratory close" in normalized
        or "when does laboratory close" in normalized
        or "what time does the lab close" in normalized
        or "what time does lab close" in normalized
        or "what time does the laboratory close" in normalized
        or "what time does laboratory close" in normalized
        or "when is the lab closed" in normalized
        or "when is the laboratory closed" in normalized

        # Generic hours
        or "lab hours" in normalized
        or "laboratory hours" in normalized
        or "lab timing" in normalized
        or "laboratory timing" in normalized
        or "lab timings" in normalized
        or "laboratory timings" in normalized
        or "working hours of the lab" in normalized
        or "working hours of lab" in normalized
        or "working hours of the laboratory" in normalized
        or "working days of the lab" in normalized
        or "working days of the laboratory" in normalized
        or "what are the lab hours" in normalized
        or "what are the laboratory hours" in normalized
        or "what are the lab timings" in normalized
        or "what are the laboratory timings" in normalized
    )

    if lab_hours_match:
        return _decision(
            Command.GET_LAB_HOURS,
            "",
        )

    # =====================================================
    # 9. LAB — LOCATION
    # =====================================================

    if _contains_any(
        normalized,
        (
            "where is the lab",
            "where is lab",
            "where is the laboratory",
            "where is laboratory",
            "where is the lab located",
            "where is the laboratory located",
            "where is the lab situated",
            "where is the laboratory situated",
            "what is the lab location",
            "what is the laboratory location",
            "lab location",
            "laboratory location",
            "location of the lab",
            "location of the laboratory",
            "where can i find the lab",
            "where can i find the laboratory",
        ),
    ):
        return _decision(
            Command.GET_LAB_LOCATION,
            "",
        )

    # =====================================================
    # 10. LAB — CONTACT
    # =====================================================

    if _contains_any(
        normalized,
        (
            "how do i contact the lab",
            "how can i contact the lab",
            "how do i contact lab",
            "how can i contact lab",
            "how do i contact the laboratory",
            "how can i contact the laboratory",
            "lab contact",
            "laboratory contact",
            "lab contact details",
            "laboratory contact details",
            "what is the lab contact",
            "what is the laboratory contact",
            "who do i contact for the lab",
            "who do i contact for the laboratory",
            "contact the lab",
            "contact the laboratory",
        ),
    ):
        return _decision(
            Command.GET_LAB_CONTACT,
            "",
        )

    # =====================================================
    # 11. LAB — EMAIL
    # =====================================================

    if _contains_any(
        normalized,
        (
            "what is the lab email",
            "what is the email of the lab",
            "what is the laboratory email",
            "what is the email of the laboratory",
            "give me the lab email",
            "give me the laboratory email",
            "lab email",
            "laboratory email",
            "lab email address",
            "laboratory email address",
            "email address of the lab",
            "email address of the laboratory",
        ),
    ):
        return _decision(
            Command.GET_LAB_EMAIL,
            "",
        )

    # =====================================================
    # 12. GENERIC LAB INFORMATION
    # =====================================================

    if _contains_any(
        normalized,
        (
            "lab information",
            "laboratory information",
            "tell me about the lab",
            "tell me about lab",
            "tell me about the laboratory",
            "tell me about laboratory",
            "lab details",
            "laboratory details",
            "information about the lab",
            "information about the laboratory",
            "details about the lab",
            "details about the laboratory",
        ),
    ):
        return _decision(
            Command.GET_LAB_INFO,
            "",
        )

    # =====================================================
    # NO DETERMINISTIC COMMAND
    # =====================================================

    return None