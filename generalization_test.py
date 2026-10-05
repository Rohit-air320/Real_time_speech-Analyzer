"""Tests the existing Wav2Vec2 emotion model on other datasets, without retraining.
Run: python generalization_test.py"""
import random
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

import config
from audio_processing import load_audio
from emotion_model import EmotionRecognizer

SAMPLES_PER_CLASS = 15  # keep this small so it runs in a few minutes; raise later if you have time

KAGGLEHUB_ROOT = Path.home() / ".cache" / "kagglehub" / "datasets"

SAVEE_CODES = {"a": "angry", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}


def collect_savee():
    root = KAGGLEHUB_ROOT / "ejlok1" / "surrey-audiovisual-expressed-emotion-savee" / "versions" / "1" / "ALL"
    items = []
    for path in root.glob("*.wav"):
        stem = path.stem  # e.g. DC_su12
        code_part = stem.split("_", 1)[1]            # su12
        code = "".join(c for c in code_part if c.isalpha())  # su
        emotion = SAVEE_CODES.get(code)
        if emotion:
            items.append((path, emotion))
    return items


def collect_tess():
    root = KAGGLEHUB_ROOT / "ejlok1" / "toronto-emotional-speech-set-tess" / "versions" / "1"
    tess_map = {"angry": "angry", "fear": "fear", "happy": "happy", "neutral": "neutral",
                "pleasant_surprise": "surprise", "sad": "sad"}
    items = []
    for folder in root.rglob("*_*"):
        if not folder.is_dir():
            continue
        label = folder.name.split("_", 1)[-1].lower()
        emotion = tess_map.get(label)
        if emotion:
            for wav in folder.glob("*.wav"):
                items.append((wav, emotion))
    return items


def collect_indic(language):
    root = (KAGGLEHUB_ROOT / "swastikatinkhede" / "multilingual-speech-dataset" / "versions" / "8"
            / "Indian Speech Emotion Dataset" / language)
    items = []
    for folder in root.glob("*"):
        if folder.is_dir():
            emotion = folder.name.lower()
            if emotion in config.EMOTIONS:
                for wav in folder.glob("*.wav"):
                    items.append((wav, emotion))
    return items


def collect_gujarati():
    root = KAGGLEHUB_ROOT / "yp271289" / "multi-language-emotion-dataset" / "versions" / "1"
    guj_map = {"angry": "angry", "fear": "fear", "fearful": "fear", "happy": "happy",
               "neutral": "neutral", "sad": "sad", "surprise": "surprise"}
    items = []
    for folder in root.rglob("*"):
        if folder.is_dir() and "gujarati" in str(folder).lower():
            emotion = guj_map.get(folder.name.lower())
            if emotion:
                for wav in folder.glob("*.wav"):
                    items.append((wav, emotion))
    return items


def evaluate(name, items, recognizer):
    if not items:
        print(f"{name}: no files found, skipping")
        return
    random.seed(0)
    by_emotion = {}
    for path, emotion in items:
        by_emotion.setdefault(emotion, []).append(path)
    sample = []
    for emotion, paths in by_emotion.items():
        random.shuffle(paths)
        sample += [(p, emotion) for p in paths[:SAMPLES_PER_CLASS]]

    y_true, y_pred = [], []
    for path, true_emotion in sample:
        try:
            audio = load_audio(str(path))
            pred_emotion, _, _ = recognizer.predict(audio)
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
            continue
        y_true.append(true_emotion)
        y_pred.append(pred_emotion)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"{name}: {len(y_true)} clips | accuracy = {acc:.3f} | macro-F1 = {f1:.3f}")


def main():
    print("Loading Wav2Vec2 model...")
    recognizer = EmotionRecognizer()

    evaluate("SAVEE (English)", collect_savee(), recognizer)
    evaluate("TESS (English)", collect_tess(), recognizer)
    evaluate("Hindi", collect_indic("Hindi"), recognizer)
    evaluate("Marathi", collect_indic("Marathi"), recognizer)
    evaluate("Gujarati", collect_gujarati(), recognizer)


if __name__ == "__main__":
    main()