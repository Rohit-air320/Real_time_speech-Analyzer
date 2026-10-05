"""Tests Whisper's transcription accuracy on Hindi and Marathi using IndicTTS samples.
Run: python test_stt_accuracy.py"""
import io

from datasets import load_from_disk, Audio
from jiwer import wer

from audio_processing import load_audio
from speech_to_text import Transcriber

SAMPLES_TO_TEST = 20
LANGUAGE_CODES = {"Hindi": "hindi", "Marathi": "marathi"}


def evaluate(name, folder, transcriber):
    print(f"\nLoading {name} dataset...")
    ds = load_from_disk(folder)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select(range(min(SAMPLES_TO_TEST, len(ds))))

    references, hypotheses = [], []
    for i, row in enumerate(ds):
        audio_entry = row["audio"]
        reference_text = row["text"].strip()

        audio = load_audio(io.BytesIO(audio_entry["bytes"]))
        result = transcriber.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={"task": "transcribe", "language": LANGUAGE_CODES[name]},
        )
        predicted_text = result["text"].strip()

        references.append(reference_text)
        hypotheses.append(predicted_text)
        print(f"  [{i+1}] ref: {reference_text[:50]}")
        print(f"       hyp: {predicted_text[:50]}")

    error_rate = wer(references, hypotheses)
    print(f"\n{name}: {len(references)} clips | Word Error Rate (WER) = {error_rate:.3f} ({error_rate*100:.1f}%)")
    return error_rate


def main():
    print("Loading Whisper model...")
    transcriber = Transcriber()

    evaluate("Hindi", "./tts_data_hindi", transcriber)
    evaluate("Marathi", "./tts_data_marathi", transcriber)


if __name__ == "__main__":
    main()