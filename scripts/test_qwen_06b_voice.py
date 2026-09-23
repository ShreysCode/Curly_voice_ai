import os
import gc
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


MODEL_PATH = r"D:\HuggingFaceCache\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REF_AUDIO = os.path.join(
    PROJECT_ROOT,
    "models",
    "curly_voice",
    "curly_reference.wav",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "audio",
)

OUTPUT_AUDIO = os.path.join(
    OUTPUT_DIR,
    "curly_qwen_test.wav",
)


# IMPORTANT:
# This must exactly match the words spoken in curly_reference.wav.
REF_TEXT = (
    "Hi, I'm Curly, created for ICAR-NRCY Centre of Excellence at ADBU. "
    "I can help you find information, explain a topic, or guide you through a task. "
    "Tell me what you need, and we'll take it one step at a time."
)


TEST_TEXT = (
    "Hi! I'm Curly. "
    "How can I help you today? "
    "I can explain information clearly and guide you through your work."
)


def main():
    print("=" * 65)
    print("CURly AI - Qwen 0.6B Voice Clone Test")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Qwen 0.6B model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(REF_AUDIO):
        raise FileNotFoundError(
            f"Curly reference audio not found:\n{REF_AUDIO}"
        )

    print("\n[1/4] Loading Qwen 0.6B Base...")
    print("Model:", MODEL_PATH)
    print("Device: CPU")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    print("\nModel loaded successfully.")

    print("\n[2/4] Creating reusable Curly voice prompt...")

    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        x_vector_only_mode=False,
    )

    print("Voice prompt created successfully.")

    print("\n[3/4] Generating Curly test speech...")
    print("Text:")
    print(TEST_TEXT)

    wavs, sample_rate = model.generate_voice_clone(
        text=TEST_TEXT,
        language="English",
        voice_clone_prompt=voice_prompt,
        max_new_tokens=2048,
    )

    if not wavs:
        raise RuntimeError("Qwen returned no audio.")

    print("\n[4/4] Saving audio...")

    sf.write(
        OUTPUT_AUDIO,
        wavs[0],
        sample_rate,
    )

    del model
    del voice_prompt
    del wavs

    gc.collect()

    print("\n" + "=" * 65)
    print("SUCCESS — QWEN CURLY VOICE TEST CREATED")
    print("=" * 65)
    print("Output:")
    print(OUTPUT_AUDIO)
    print("Sample rate:", sample_rate)
    print("=" * 65)


if __name__ == "__main__":
    main()
