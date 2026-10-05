"""Loading / cleaning audio and reading the RAVDESS dataset."""
from pathlib import Path

import librosa
import numpy as np

import config


def preprocess(audio: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence, normalise volume, cap the length."""
    audio = audio.astype(np.float32)
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    if trimmed.size > 0:
        audio = trimmed
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio[: config.SAMPLE_RATE * config.MAX_SECONDS]


def load_audio(source) -> np.ndarray:
    """Load a file path or file-like object as mono 16 kHz audio and preprocess it."""
    audio, _ = librosa.load(source, sr=config.SAMPLE_RATE, mono=True)
    return preprocess(audio)


def load_ravdess(dataset_dir=config.DATASET_DIR):
    """Return a list of {'path', 'emotion', 'actor'} for every usable RAVDESS file."""
    items = []
    for path in sorted(Path(dataset_dir).rglob("*.wav")):
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue
        emotion = config.RAVDESS_EMOTIONS.get(parts[2])
        if emotion is None:
            continue
        items.append({"path": str(path), "emotion": emotion, "actor": int(parts[6])})
    return items


def split_by_actor(items):
    """Speaker-independent train / validation / test split."""
    train = [i for i in items if i["actor"] in config.TRAIN_ACTORS]
    val = [i for i in items if i["actor"] in config.VAL_ACTORS]
    test = [i for i in items if i["actor"] in config.TEST_ACTORS]
    return train, val, test
