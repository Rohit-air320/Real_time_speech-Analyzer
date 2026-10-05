"""Speech-to-text with Whisper (Hugging Face Transformers)."""
import sys

import torch
from transformers import pipeline

import config


class Transcriber:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL,
            device=device,
            chunk_length_s=30,
        )

    def transcribe(self, audio) -> str:
        result = self.pipe(
    {"raw": audio, "sampling_rate": config.SAMPLE_RATE},
    generate_kwargs={"task": "translate"},   # was "transcribe"
)
        return result["text"].strip()


if __name__ == "__main__":  # python speech_to_text.py some_file.wav
    from audio_processing import load_audio

    print(Transcriber().transcribe(load_audio(sys.argv[1])))
