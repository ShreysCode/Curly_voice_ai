import os
import av
import tempfile
import time
import wave
from pathlib import Path
import re

import numpy as np
import sounddevice as sd

from app.core.curly import Curly, CurlyState
from app.models.schemas import (
    Command, 
    CurlyResponse,
    CurlyState,
    IntentSource,
    ResponseType,
)
from app.stt.service import STTService
from app.tts.service import TTSService


class CurlyVoiceEngine:
    """Mac development voice harness for Curly.

    Wake-word detection intentionally does not live here. Android will own
    wake-word detection in the production client.
    """

    def __init__(
        self,
        curly: Curly,
        stt: STTService,
        tts: TTSService,
        sample_rate: int = 16000,
        silence_seconds: float = 0.8,
        max_record_seconds: float = 10.0,
        session_timeout_seconds: int = 300,
    ):
        self.curly = curly
        self.stt = stt
        self.tts = tts
        self.sample_rate = sample_rate
        self.silence_seconds = silence_seconds
        self.max_record_seconds = max_record_seconds
        self.session_timeout_seconds = session_timeout_seconds

        self.session_id = self.curly.create_session()
        self.last_activity = time.monotonic()

    # --------------------------------------------------
    # AUDIO RECORDING
    # --------------------------------------------------

    def record_audio_until_silence(
        self,
        max_duration: float | None = None,
        silence_duration: float | None = None,
        block_duration: float = 0.1,
        threshold: float = 0.015,
        speech_blocks_required: int = 2,
    ) -> Path:
        """Record one utterance after the user manually starts listening.

        This is deliberately simple on Mac because it is now push-to-talk.
        Background listening / wake-word detection is not part of this path.
        """
        max_duration = (
            self.max_record_seconds
            if max_duration is None
            else max_duration
        )
        silence_duration = (
            self.silence_seconds
            if silence_duration is None
            else silence_duration
        )

        block_size = int(self.sample_rate * block_duration)
        max_blocks = max(1, int(max_duration / block_duration))
        silence_blocks_required = max(
            1,
            int(silence_duration / block_duration),
        )

        frames: list[np.ndarray] = []
        speech_started = False
        consecutive_speech_blocks = 0
        silent_blocks = 0

        print("Listening...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:
            for _ in range(max_blocks):
                audio, _ = stream.read(block_size)
                audio = audio.reshape(-1)

                energy = float(np.sqrt(np.mean(audio * audio)))
                is_speech_like = energy >= threshold

                if not speech_started:
                    if is_speech_like:
                        consecutive_speech_blocks += 1
                        frames.append(audio.copy())
                        if consecutive_speech_blocks >= speech_blocks_required:
                            speech_started = True
                            silent_blocks = 0
                    else:
                        consecutive_speech_blocks = 0
                        continue

                else:
                    frames.append(audio.copy())

                    if is_speech_like:
                        silent_blocks = 0
                    else:
                        silent_blocks += 1
                        if silent_blocks >= silence_blocks_required:
                            break

        if not frames:
            recording = np.zeros(1, dtype=np.int16)
        else:
            audio_data = np.concatenate(frames)
            recording = np.clip(
                audio_data * 32767,
                -32768,
                32767,
            ).astype(np.int16)

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.close()

        with wave.open(temp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(recording.tobytes())

        return Path(temp.name)

    # --------------------------------------------------
    # SPEECH PLAYBACK
    # --------------------------------------------------

    def play_audio(self, audio_path: Path) -> None:
        """Play an audio file cross-platform using PyAV + sounddevice."""

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        container = av.open(str(audio_path))

        try:
            audio_stream = next(
                (
                    stream
                    for stream in container.streams
                    if stream.type == "audio"
                ),
                None,
            )

            if audio_stream is None:
                raise RuntimeError(
                    f"No audio stream found in {audio_path}"
                )

            sample_rate = audio_stream.rate or 24000

            # Convert whatever Edge TTS produced into
            # signed 16-bit, mono PCM for sounddevice.
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=sample_rate,
            )

            chunks: list[np.ndarray] = []

            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)

                if resampled_frames is None:
                    continue

                if not isinstance(
                    resampled_frames,
                    (list, tuple),
                ):
                    resampled_frames = [resampled_frames]

                for resampled in resampled_frames:
                    if resampled is None:
                        continue

                    audio = resampled.to_ndarray()

                    # Mono audio may arrive as shape (1, samples).
                    audio = audio.reshape(-1)

                    chunks.append(audio)

            # Flush any remaining resampler samples.
            flushed_frames = resampler.resample(None)

            if flushed_frames is not None:
                if not isinstance(
                    flushed_frames,
                    (list, tuple),
                ):
                    flushed_frames = [flushed_frames]

                for resampled in flushed_frames:
                    if resampled is None:
                        continue

                    audio = resampled.to_ndarray()
                    audio = audio.reshape(-1)
                    chunks.append(audio)

            if not chunks:
                raise RuntimeError(
                    f"No decoded audio samples found in {audio_path}"
                )

            pcm = np.concatenate(chunks)

            sd.play(
                pcm,
                samplerate=sample_rate,
            )

            sd.wait()

        finally:
            container.close()

    async def speak(self, text: str) -> None:
        if not text:
            return

        audio_path = await self.tts.synthesize(text)
        try:
            self.play_audio(audio_path)
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    # --------------------------------------------------
    # STT
    # --------------------------------------------------

    def transcribe(self, audio_path: Path) -> str:
        result = self.stt.transcribe(str(audio_path))
        if not result:
            return ""
        return result.get("text", "").strip()

    # --------------------------------------------------
    # PROCESS USER REQUEST
    # --------------------------------------------------

    async def process_request(self, text: str) -> CurlyResponse:
        self.last_activity = time.monotonic()

        print("\nCurly: ", end="", flush=True)

        # -----------------------------
        # CURLY / LLM
        # -----------------------------
        curly_start = time.perf_counter()

        chunks: list[str] = []

        async for token in self.curly.stream_normal_response(
            session_id=self.session_id,
            text=text,
        ):
            if token:
                chunks.append(token)
                print(token, end="", flush=True)

        curly_time = time.perf_counter() - curly_start
        print()

        response_text = "".join(chunks).strip()

        if not response_text:
            response_text = "I didn't get a response."

        print(f"[TIMING] Curly/LLM: {curly_time:.2f}s")
        print(f"[TIMING] Response words: {len(response_text.split())}")

        # -----------------------------
        # TTS
        # -----------------------------
        tts_start = time.perf_counter()

        audio_path = await self.tts.synthesize(response_text)

        tts_time = time.perf_counter() - tts_start

        print(f"[TIMING] TTS generation: {tts_time:.2f}s")

        # -----------------------------
        # PLAYBACK
        # -----------------------------
        playback_start = time.perf_counter()

        self.play_audio(audio_path)

        playback_time = time.perf_counter() - playback_start

        print(f"[TIMING] Playback: {playback_time:.2f}s")

        audio_path.unlink(missing_ok=True)

        total_time = curly_time + tts_time + playback_time

        print(
            "\n========== CURLY TIMING ==========\n"
            f"Curly:    {curly_time:.2f}s\n"
            f"TTS:      {tts_time:.2f}s\n"
            f"Playback: {playback_time:.2f}s\n"
            f"TOTAL:    {total_time:.2f}s\n"
            "=================================="
        )

        self.curly.set_state(
            self.session_id,
            CurlyState.SPEAKING,
        )

        return CurlyResponse(
            type=ResponseType.RESPONSE,
            command=Command.NONE,
            text=response_text,
            state=CurlyState.SPEAKING,
            intent_source=IntentSource.LLM,
        )

    # --------------------------------------------------
    # AUTHENTICATION RESULT
    # --------------------------------------------------

    async def handle_auth_result(
        self,
        status: str,
        name: str | None = None,
    ):
        response = self.curly.handle_auth_result(
            self.session_id,
            {
                "status": status,
                "name": name,
            },
        )

        print(f"Curly: {response.text}")
        await self.speak(response.text)

        self.last_activity = time.monotonic()
        self.curly.set_state(
            self.session_id,
            CurlyState.LISTENING,
        )

    # --------------------------------------------------
    # PUSH-TO-TALK LOOP
    # --------------------------------------------------

    async def run(self):
        print()
        print("================================")
        print("        CURLY VOICE ENGINE")
        print("         MAC TEST MODE")
        print("================================")
        print()
        print("Wake-word detection is disabled on Mac.")
        print("Press ENTER to talk to Curly.")
        print("Type q + ENTER to quit.")
        print()

        self.curly.set_state(self.session_id, CurlyState.IDLE)

        while True:
            try:
                command = input("\n[Press ENTER to speak | q to quit] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting Curly voice test.")
                break

            if command == "q":
                print("Exiting Curly voice test.")
                break

            if command:
                print("Press ENTER without typing anything, or q to quit.")
                continue

            self.curly.set_state(self.session_id, CurlyState.LISTENING)

            started = time.perf_counter()

            request_audio = self.record_audio_until_silence()
            record_time = time.perf_counter() - started

            try:
                stt_started = time.perf_counter()
                try:
                    text = self.transcribe(request_audio)
                except ValueError:
                    print("Curly: Please speak in English.")
                    await self.speak("Please speak in English.")
                    continue
                stt_time = time.perf_counter() - stt_started
            except RuntimeError as error:
                print(f"STT error: {error}")
                continue
            finally:
                request_audio.unlink(missing_ok=True)

            if not text:
                print("[No speech detected]")
                continue

            print(f"You: {text}")

            try:
                llm_started = time.perf_counter()
                response = await self.process_request(text)
                processing_time = time.perf_counter() - llm_started
            except Exception as error:
                print(f"Curly processing error: {error}")
                self.curly.set_state(
                    self.session_id,
                    CurlyState.ERROR,
                )
                try:
                    await self.speak(
                        "I'm having trouble connecting to my AI system."
                    )
                except Exception:
                    pass
                continue

            print(f"Curly: {response.text}")

            total_time = (
                record_time
                + stt_time
                + processing_time
            )

            print(
                "\n"
                "========== TURN TIMING ==========\n"
                f"Record:     {record_time:.2f}s\n"
                f"STT:        {stt_time:.2f}s\n"
                f"Curly/TTS:  {processing_time:.2f}s\n"
                f"TOTAL:      {total_time:.2f}s\n"
                "================================="
            )

            if response.command == Command.FACE_AUTH:
                print()
                print("FACE_AUTH requested.")
                print("Android will own the real face-authentication flow.")
                print()
                self.curly.set_state(
                    self.session_id,
                    CurlyState.AUTHENTICATING,
                )
                continue

            if response.state == CurlyState.IDLE:
                self.curly.set_state(
                    self.session_id,
                    CurlyState.IDLE,
                )
