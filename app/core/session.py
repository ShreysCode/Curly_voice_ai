from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Session:

    session_id: str

    history: list[dict[str, str]] = field(
        default_factory=list
    )

    state: str = "IDLE"

    created_at: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    last_activity: datetime = field(
        default_factory=lambda:
        datetime.now(timezone.utc)
    )

    def touch(self):

        self.last_activity = (
            datetime.now(timezone.utc)
        )

    def set_state(self, state):

        self.state = state
        self.touch()

    def is_expired(
        self,
        timeout_seconds: int
    ) -> bool:

        now = datetime.now(
            timezone.utc
        )

        age = (
            now - self.last_activity
        ).total_seconds()

        return age > timeout_seconds