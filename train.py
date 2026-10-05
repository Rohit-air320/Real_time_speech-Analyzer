"""Train the SVM baseline on RAVDESS (actors 1-16), choosing C on actors 17-20.
Run:  python train.py
"""
import numpy as np
from sklearn.metrics import accuracy_score

import config
from audio_processing import load_audio, load_ravdess, split_by_actor
from baseline_model import build_svm, extract_features, save_model


def featurize(items):
    X = np.stack([extract_features(load_audio(item["path"])) for item in items])
    y = np.array([item["emotion"] for item in items])
    return X, y


def main():
    items = load_ravdess()
    if not items:
        raise SystemExit(f"No RAVDESS .wav files found in {config.DATASET_DIR}. See README.md, step 3.")
    train, val, _ = split_by_actor(items)
    print(f"Files: {len(items)} | train {len(train)} | validation {len(val)}")

    print("Extracting features (takes a few minutes)...")
    X_train, y_train = featurize(train)
    X_val, y_val = featurize(val)

    best_model, best_acc = None, -1.0
    for C in (1, 10, 100):
        model = build_svm(C).fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"C = {C:<4} validation accuracy = {acc:.3f}")
        if acc > best_acc:
            best_model, best_acc = model, acc

    save_model(best_model)
    print(f"Saved best model to {config.SVM_PATH} (validation accuracy {best_acc:.3f})")


if __name__ == "__main__":
    main()
