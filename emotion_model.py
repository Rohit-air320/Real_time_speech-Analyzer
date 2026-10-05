"""Emotion recognition from the voice with a pretrained Wav2Vec2 model."""
import sys

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

import config


class ClassificationHead(nn.Module):
    """The emotion layer stored inside the downloaded model: dense -> tanh -> out_proj."""

    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.dropout = nn.Dropout(model_config.final_dropout)
        self.out_proj = nn.Linear(model_config.hidden_size, model_config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForEmotion(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 + average over time + the classification head above."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.wav2vec2 = Wav2Vec2Model(model_config)
        self.classifier = ClassificationHead(model_config)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values).last_hidden_state   # (1, frames, hidden)
        return self.classifier(hidden.mean(dim=1))               # (1, labels)


def load_emotion_model():
    model, info = Wav2Vec2ForEmotion.from_pretrained(config.EMOTION_MODEL, output_loading_info=True)
    # Safety check: if the emotion layer did not load, predictions would be random.
    missing = [k for k in info.get("missing_keys", []) if k.startswith("classifier")]
    if missing:
        unused = [k for k in info.get("unexpected_keys", []) if "classifier" in k]
        raise RuntimeError(
            "The emotion layer of the model did not load correctly.\n"
            f"Missing: {missing}\nUnused in file: {unused}\n"
            "Send this message to your guide/assistant."
        )
    return model


class EmotionRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.EMOTION_MODEL)
        self.model = load_emotion_model().to(self.device).eval()

        names = self.model.config.id2label
        labels = [str(names[i]).lower() for i in range(len(names))]
        if all(name.startswith("label_") for name in labels):   # config had no real names
            labels = config.MODEL_LABELS_FALLBACK
        self.labels = labels

    @torch.no_grad()
    def predict(self, audio):
        """Return (emotion, confidence, {emotion: probability}) over the six project emotions."""
        inputs = self.extractor(
            audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", padding=True
        )
        logits = self.model(inputs.input_values.to(self.device))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        raw = {name: float(p) for name, p in zip(self.labels, probs) if name in config.EMOTIONS}
        total = sum(raw.values()) or 1.0
        scores = {e: raw.get(e, 0.0) / total for e in config.EMOTIONS}
        emotion = max(scores, key=scores.get)
        return emotion, scores[emotion], scores


if __name__ == "__main__":  # python emotion_model.py some_file.wav
    from audio_processing import load_audio

    label, conf, all_scores = EmotionRecognizer().predict(load_audio(sys.argv[1]))
    print(label, f"{conf:.2%}")
    for name, value in all_scores.items():
        print(f"  {name:9s} {value:.2%}")