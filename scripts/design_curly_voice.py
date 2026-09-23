import os

# Keep all Hugging Face/Qwen model cache on D:
os.environ["HF_HOME"] = r"D:\HuggingFaceCache"
os.environ["HF_HUB_CACHE"] = r"D:\HuggingFaceCache\hub"

import gc
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


MODEL_ID = r"D:\HuggingFaceCache\hub\models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign\snapshots\5ecdb67327fd37bb2e042aab12ff7391903235d3"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "curly_voice",
)

OUTPUT_WAV = os.path.join(
    OUTPUT_DIR,
    "curly_reference_v2.wav",
)

OFFLOAD_DIR = r"D:\QwenOffload"


REFERENCE_TEXT = (
    "Hi, I'm Curly, created for ICAR-NRCY Centre of Excellence at ADBU. "
    "I can help you find information, explain a topic, "
    "or guide you through a task. "
    "Tell me what you need, and we'll take it one step at a time."
)


VOICE_DESCRIPTION = (
    "A natural human female voice with a youthful young-teen to late-teen "
    "impression, around 11 to 13years old. "
    "She speaks natural Indian English with a subtle Indian accent. "
    "The voice should sound like a real young Indian woman speaking naturally "
    "in a quiet indoor conversation or recording studio. "
    "Warm, clear, soft and youthful, but grounded and realistic. "
    "Use a natural female pitch with a comfortable mid-to-slightly-high range. "
    "Keep the pitch stable and realistic rather than exaggerated. "
    "The delivery should be conversational, relaxed, confident and articulate. "
    "She should sound intelligent, approachable and professional, "
    "with gentle friendliness and a small amount of natural energy. "
    "Use realistic human breathing, natural pauses and subtle intonation. "
    "Keep emotional expression restrained and believable. "
    "The overall impression should be an actual human young woman, "
    "not a fictional character. "
    "Do not make the voice cute in a childish way. "
    "Do not make it sound like anime, animation, gaming, dubbing, "
    "cartoon, mascot, fantasy character, virtual idol or exaggerated AI voice. "
    "Avoid exaggerated pitch changes, squeaky tones, theatrical acting, "
    "baby-like speech, childish pronunciation, sing-song delivery, "
    "overly energetic delivery, robotic delivery, monotone speech, "
    "breathy ASMR style, or dramatic character performance."
)


def main():
    print("=" * 65)
    print("CURly AI - Qwen3-TTS Human-Centric VoiceDesign")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    print("\n[1/4] Checking environment...")
    print("Python:", os.sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    print("\n[2/4] Loading Qwen VoiceDesign model...")
    print("Model:", MODEL_ID)
    print("Offload:", OFFLOAD_DIR)

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_folder=OFFLOAD_DIR,
        attn_implementation="sdpa",
    )

    print("\nModel loaded successfully.")

    print("\n[3/4] Generating human-centric Curly reference voice...")

    print("\nReference text:")
    print(REFERENCE_TEXT)

    print("\nVoice description:")
    print(VOICE_DESCRIPTION)

    wavs, sample_rate = model.generate_voice_design(
        text=REFERENCE_TEXT,
        language="English",
        instruct=VOICE_DESCRIPTION,
        max_new_tokens=2048,
    )

    if not wavs:
        raise RuntimeError("Qwen returned no audio.")

    print("\n[4/4] Saving audio...")

    sf.write(
        OUTPUT_WAV,
        wavs[0],
        sample_rate,
    )

    del model
    del wavs

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print("SUCCESS — HUMAN-CENTRIC CURLY VOICE CREATED")
    print("=" * 65)
    print("File:")
    print(OUTPUT_WAV)
    print("Sample rate:", sample_rate)
    print("=" * 65)


if __name__ == "__main__":
    main()