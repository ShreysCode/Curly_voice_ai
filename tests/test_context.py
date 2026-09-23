import pytest
import json

from app.core.curly import Curly


class ContextLLM:

    def __init__(self):
        self.calls = []
        self.response_calls = []

    async def generate(
     self,
     messages,
     response_format=None
    ):

      self.calls.append(messages)

      user_message = messages[-1]["content"].lower()

      if "who is the lab in-charge" in user_message:

        return json.dumps({
            "type": "response",
            "command": "NONE",
            "text": (
                "Dr. Example is the laboratory in-charge."
            ),
            "confidence": 1.0
        })

      if (
        "what department"
        in user_message
        and any(
            "Dr. Example"
            in message["content"]
            for message in messages
            if message["role"] == "assistant"
        )
    ):

        return json.dumps({
            "type": "response",
            "command": "NONE",
            "text": (
                "They are from the Example Department."
            ),
            "confidence": 1.0
        })

      return json.dumps({
        "type": "response",
        "command": "NONE",
        "text": "I don't have that information.",
        "confidence": 1.0
    })

    async def health(self):
        return True

class FakeKnowledge:

    def get_context(self, topic=None):

        return """
        {
            "lab": {
                "incharge": "Dr. Example",
                "department": "Example Department"
            }
        }
        """


@pytest.mark.asyncio
async def test_multi_turn_context():

    llm = ContextLLM()

    curly = Curly(
        llm=llm,
        knowledge=FakeKnowledge()
    )

    session_id = curly.create_session()

    first = await curly.chat(
        session_id,
        "Who is the lab in-charge?"
    )

    assert "Dr. Rupesh Mandal" in first.text

    second = await curly.chat(
        session_id,
        "Who is the lab in-charge again?"
    )

    assert "Dr. Rupesh Mandal" in second.text