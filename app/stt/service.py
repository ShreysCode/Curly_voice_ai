from faster_whisper import WhisperModel

from app.config import settings


class STTService:

    def __init__(self):

        print(
            f"Loading Whisper model: {settings.stt_model}"
        )

        self.model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )

        print("Whisper model loaded.")

    def transcribe(
        self,
        audio_path: str
    ) -> dict:

        try:

            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                beam_size=5,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text
            ).strip()

            detected_language = (
                info.language
                if info.language
                else "en"
            )

            language_probability = (
                float(info.language_probability)
                if info.language_probability is not None
                else 0.0
            )

            # -----------------------------------------
            # ENGLISH-ONLY CHECK
            # -----------------------------------------

            if (
                detected_language != "en"
                or language_probability < 0.70
            ):
                raise ValueError(
                    "Please speak in English."
                )

            return {
                "text": text,
                "language": detected_language,
                "language_probability": (
                    language_probability
                ),
            }

        except ValueError:
            raise

        except Exception as error:

            raise RuntimeError(
                "Speech recognition failed."
            ) from error