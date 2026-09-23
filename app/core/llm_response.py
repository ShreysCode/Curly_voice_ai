import json

from app.core.prompts import SYSTEM_PROMPT
from app.llm.ollama import OllamaClient
from app.models.schemas import (
    Command,
    LLMDecision,
    ResponseType,
)


LLM_RESPONSE_PROMPT = """
You are Curly's fast decision and response generator.

Return ONLY valid JSON.

Commands:
- NONE
- FACE_AUTH
- GET_TIME
- GET_WEATHER
- GET_LAB_INFO
- END_CONVERSATION

Rules:
1. Use NONE for normal conversation.
2. Use FACE_AUTH only when the user asks to authenticate, verify,
   scan, identify, or check themselves for access.
3. Use GET_TIME for current-time questions.
4. Use GET_WEATHER for current-weather questions.
5. Use GET_LAB_INFO for institutional laboratory questions.
6. Use END_CONVERSATION only when the user clearly wants to end the session.
7. Never decide whether a person is authorized.
8. Never invent current time or weather.
9. Use supplied knowledge for institutional questions.
10. Resolve simple references such as "they", "he", "she", "it", or "that"
    using the recent conversation history.
11. Keep spoken responses concise: normally 1–3 short sentences.
12. If the supplied knowledge does not contain an institutional fact,
    say: "I don't have that information yet."
13. Confidence must be between 0 and 1.

Return exactly:

{
  "type": "response" or "command",
  "command": "NONE" or one of the commands above,
  "text": "short spoken response",
  "confidence": 0.0
}
"""


class LLMResponseGenerator:

    def __init__(
        self,
        llm: OllamaClient,
    ):
        self.llm = llm

    @staticmethod
    def _compact_history(
        history: list[dict[str, str]],
        max_messages: int = 6,
    ) -> list[dict[str, str]]:
        """
        Keep only the most recent conversation messages.

        Six messages = roughly the last three user/assistant turns.
        This is enough for normal pronoun/reference resolution while
        preventing the prompt from growing indefinitely.
        """

        if not history:
            return []

        compacted = history[-max_messages:]

        result: list[dict[str, str]] = []

        for message in compacted:

            role = message.get("role")

            content = message.get("content", "").strip()

            if role not in {
                "user",
                "assistant",
            }:
                continue

            if not content:
                continue

            result.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return result

    async def generate(
        self,
        user_text: str,
        history: list[dict[str, str]],
        knowledge_context: str = "",
        environment_context: dict | None = None,
    ) -> LLMDecision:

        recent_history = self._compact_history(
            history
        )

        # -----------------------------------------
        # Keep the actual prompt compact.
        # -----------------------------------------

        prompt_parts = [
            "KNOWLEDGE:",
            knowledge_context.strip()
            if knowledge_context
            else "None",
        ]

        if environment_context:
            prompt_parts.extend(
                [
                    "",
                    "ENVIRONMENT:",
                    str(environment_context),
                ]
            )

        prompt_parts.extend(
            [
                "",
                "USER:",
                user_text.strip(),
            ]
        )

        prompt = "\n".join(
            prompt_parts
        )

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\n"
                    + LLM_RESPONSE_PROMPT
                ),
            }
        ]

        messages.extend(
            recent_history
        )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        response_schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "response",
                        "command",
                    ],
                },
                "command": {
                    "type": "string",
                    "enum": [
                        "NONE",
                        "FACE_AUTH",
                        "GET_TIME",
                        "GET_WEATHER",
                        "GET_LAB_INFO",
                        "END_CONVERSATION",
                    ],
                },
                "text": {
                    "type": "string",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "type",
                "command",
                "text",
                "confidence",
            ],
        }

        try:

            response = await self.llm.generate(
                messages=messages,
                response_format=response_schema,
            )

            data = json.loads(
                response
            )

            return LLMDecision(
                **data
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ):

            return LLMDecision(
                type=ResponseType.RESPONSE,
                command=Command.NONE,
                text=(
                    "I'm sorry, I didn't quite "
                    "understand that."
                ),
                confidence=0.0,
            )