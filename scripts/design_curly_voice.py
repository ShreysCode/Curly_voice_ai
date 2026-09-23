import os
import gc
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


MODEL_PATH = (
    r"D:\HuggingFaceCache\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
    r"\snapshots\5d83992436eae1d760afd27aff78a71d676296fc"
)

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


# EXACT transcript of curly_reference.wav
REF_TEXT = (
    "Hey! I'm Curly. I'm here to help you with your questions "
    "about the ICAR National Research Centre on Yak. Let's get started!"
)

TEST_TEXT = (
    "Hi! I'm Curly. How can I help you today? "
    "I can explain information clearly and guide you through your work."
)


def main():
    print("=" * 65)
    print("CURly AI - Qwen3-TTS 0.6B Voice Clone")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"\n0.6B model not found:\n{MODEL_PATH}"
        )

    if not os.path.isfile(REF_AUDIO):
        raise FileNotFoundError(
            f"\nCurly reference audio not found:\n{REF_AUDIO}"
        )

    print("\n[1/3] Loading Qwen 0.6B Base model...")
    print("Model:", MODEL_PATH)
    print("Device: CPU")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    print("Model loaded successfully.")

    print("\n[2/3] Generating speech using Curly reference...")
    print("Reference transcript:")
    print(REF_TEXT)
    print("\nTest text:")
    print(TEST_TEXT)

    wavs, sample_rate = model.generate_voice_clone(
        text=TEST_TEXT,
        language="English",
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        x_vector_only_mode=False,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    if not wavs:
        raise RuntimeError(
            "Qwen returned no audio."
        )

    print("\n[3/3] Saving generated audio...")

    sf.write(
        OUTPUT_AUDIO,
        wavs[0],
        sample_rate,
    )

    del model
    del wavs

    gc.collect()

    print("\n" + "=" * 65)
    print("SUCCESS")
    print("=" * 65)
    print("Curly test audio:")
    print(OUTPUT_AUDIO)
    print("Sample rate:", sample_rate)
    print("=" * 65)


if __name__ == "__main__":
    main()