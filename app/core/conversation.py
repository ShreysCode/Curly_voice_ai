import time
from dataclasses import dataclass


@dataclass
class ConversationSession:

    session_id: str

    active: bool = False

    last_activity: float = 0.0

    timeout_seconds: int = 300

    def activate(self):

        self.active = True
        self.touch()

    def deactivate(self):

        self.active = False

    def touch(self):

        self.last_activity = time.monotonic()

    def is_expired(self) -> bool:

        if not self.active:
            return True

        return (
            time.monotonic()
            - self.last_activity
            > self.timeout_seconds
        )