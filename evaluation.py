"""Evaluate the SVM baseline and the Wav2Vec2 model on the test actors (21-24).
Saves numbers and graphs in results/.   Run:  python evaluation.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

import config
from audio_processing import load_audio, load_ravdess, split_by_actor
from baseline_model import extract_features, load_model
from emotion_model import EmotionRecognizer


def main():
    _, _, test = split_by_actor(load_ravdess())
    if not test:
        raise SystemExit(f"No test files found in {config.DATASET_DIR}. See README.md, step 3.")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = [item["emotion"] for item in test]
    print(f"Test files: {len(test)} (actors {config.TEST_ACTORS.start}-{config.TEST_ACTORS.stop - 1})")

    print("SVM baseline...")
    svm = load_model()
    X_test = np.stack([extract_features(load_audio(item["path"])) for item in test])
    predictions = {"SVM baseline": list(svm.predict(X_test))}

    print("Wav2Vec2 model...")
    recognizer = EmotionRecognizer()
    predictions["Wav2Vec2"] = [recognizer.predict(load_audio(item["path"]))[0] for item in test]

    metrics = {}
    for name, y_pred in predictions.items():
        metrics[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=config.EMOTIONS, average="macro"),
            "report": classification_report(
                y_true, y_pred, labels=config.EMOTIONS, output_dict=True, zero_division=0
            ),
        }
        print(f"\n=== {name} ===  accuracy {metrics[name]['accuracy']:.3f} | macro-F1 {metrics[name]['macro_f1']:.3f}")
        print(classification_report(y_true, y_pred, labels=config.EMOTIONS, zero_division=0))

        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=config.EMOTIONS, xticks_rotation=45, cmap="Blues"
        )
        plt.title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()

    with open(config.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    names = list(metrics)
    x = np.arange(2)
    width = 0.35
    for i, name in enumerate(names):
        plt.bar(x + i * width, [metrics[name]["accuracy"], metrics[name]["macro_f1"]], width, label=name)
    plt.xticks(x + width / 2, ["Accuracy", "Macro-F1"])
    plt.ylim(0, 1)
    plt.title("Model comparison on test actors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved metrics.json and graphs in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
