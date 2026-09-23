import json

from app.llm.ollama import OllamaClient
from app.models.schemas import Command


INTENT_SYSTEM_PROMPT = """
You are Curly's intent classification system.

Your ONLY job is to determine whether the user's message represents
one of Curly's supported application commands.

Available commands:

NONE
FACE_AUTH
GET_TIME
GET_WEATHER
GET_LAB_INFO
END_CONVERSATION

Definitions:

FACE_AUTH:
The user wants Curly to verify, authenticate, identify, scan, or check
their identity for access.

GET_TIME:
The user wants to know the current time.

GET_WEATHER:
The user wants current weather information.

GET_LAB_INFO:
The user is asking for information about the laboratory.

END_CONVERSATION:
The user wants to end the current Curly conversation.

NONE:
Normal conversation or anything that does not clearly represent one
of the above commands.

IMPORTANT:

Never infer FACE_AUTH merely because the user mentions doors,
security, identity, or access in a general discussion.

FACE_AUTH should only be selected when the user is actually asking
Curly to perform or initiate authentication.

Return ONLY valid JSON.

Format:

{
    "command": "NONE",
    "confidence": 0.0
}

confidence must be between 0.0 and 1.0.
"""


class LLMIntentClassifier:

    def __init__(
        self,
        llm: OllamaClient
    ):
        self.llm = llm

    async def classify(
        self,
        text: str
    ) -> tuple[Command, float]:

        messages = [
            {
                "role": "system",
                "content": INTENT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]

        response = await self.llm.generate(
            messages=messages,
            response_format={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "NONE",
                            "FACE_AUTH",
                            "GET_TIME",
                            "GET_WEATHER",
                            "GET_LAB_INFO",
                            "END_CONVERSATION"
                        ]
                    },
                    
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": [
                    "command",
                    "confidence"
                ]
            }
        )

        try:

            data = json.loads(response)

            command = Command(
                data["command"]
            )

            confidence = float(
                data["confidence"]
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            return command, confidence

        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError
        ):

            return Command.NONE, 0.0