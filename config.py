"""Central settings. Change values here, not inside the other files."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ---------- Audio ----------
SAMPLE_RATE = 16000      # Whisper and Wav2Vec2 both expect 16 kHz
MAX_SECONDS = 15         # longer recordings are cut to keep the demo fast
MIN_SECONDS = 0.5        # shorter recordings are rejected

# ---------- Emotions used in the whole project ----------
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]

# ---------- Models (all free / open-source) ----------
WHISPER_MODEL = "openai/whisper-medium"
EMOTION_MODEL = "r-f/wav2vec-english-speech-emotion-recognition"
# This model also predicts "disgust". We drop it and re-normalise the other six scores.
MODEL_LABELS_FALLBACK = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------- LLM analysis ----------
# "local"  -> free, runs on your computer (downloads a small model once)
# "openai" -> uses the OpenAI API (needs the OPENAI_API_KEY environment variable, paid)
LLM_BACKEND = "local"
LOCAL_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
OPENAI_MODEL = "gpt-4o-mini"

# ---------- RAVDESS (dataset/ folder) ----------
# Speaker-independent split: no actor appears in more than one set.
TRAIN_ACTORS = range(1, 17)    # actors 1-16
VAL_ACTORS = range(17, 21)     # actors 17-20
TEST_ACTORS = range(21, 25)    # actors 21-24

# RAVDESS file name = modality-channel-EMOTION-intensity-statement-repetition-ACTOR.wav
# emotion code -> our label. "calm" is merged into "neutral", "disgust" is not used.
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": None, "08": "surprise",
}

SVM_PATH = MODELS_DIR / "svm_baseline.joblib"
