import uuid
from pathlib import Path

import edge_tts

from app.config import settings


class TTSService:

    def __init__(
        self,
        output_dir: str = "audio"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    async def synthesize(
        self,
        text: str
    ) -> Path:

        if not text.strip():
            raise ValueError(
                "TTS text cannot be empty"
            )

        filename = (
            f"{uuid.uuid4().hex}.mp3"
        )

        output_path = (
            self.output_dir / filename
        )

        communicator = edge_tts.Communicate(
            text=text,
            voice=settings.tts_voice,
            rate=settings.tts_rate,
            volume=settings.tts_volume
        )

        await communicator.save(
            str(output_path)
        )

        return output_path