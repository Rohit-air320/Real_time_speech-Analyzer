"""Baseline: hand-made audio features + SVM (to compare against Wav2Vec2)."""
import joblib
import librosa
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def extract_features(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """MFCC, pitch, energy, zero-crossing rate and spectral centroid (mean and std of each)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pitch = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    energy = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    features = list(mfcc.mean(axis=1)) + list(mfcc.std(axis=1))
    for series in (pitch, energy, zcr, centroid):
        features += [float(np.mean(series)), float(np.std(series))]
    return np.array(features, dtype=np.float32)


def build_svm(C: float = 10.0):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C))


def save_model(model, path=config.SVM_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=config.SVM_PATH):
    return joblib.load(path)
