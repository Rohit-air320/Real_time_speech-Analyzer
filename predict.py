"""The full pipeline: audio -> text -> emotion -> LLM analysis."""
import sys
import time

import config
from emotion_model import EmotionRecognizer
from llm_analysis import LLMAnalyzer
from speech_to_text import Transcriber


class SpeechEmotionPipeline:
    def __init__(self):
        self.stt = Transcriber()
        self.ser = EmotionRecognizer()
        self.llm = LLMAnalyzer()

    def run(self, audio) -> dict:
        if len(audio) < config.SAMPLE_RATE * config.MIN_SECONDS:
            raise ValueError("The recording is too short or silent. Please speak for a second or more.")

        timings = {}

        start = time.time()
        text = self.stt.transcribe(audio)
        timings["speech-to-text"] = time.time() - start

        start = time.time()
        emotion, confidence, scores = self.ser.predict(audio)
        timings["emotion recognition"] = time.time() - start

        start = time.time()
        if text:
            analysis = self.llm.analyze(text, emotion, confidence, scores)
        else:
            analysis = "No speech was recognised, so there is nothing to analyse."
        timings["LLM analysis"] = time.time() - start

        timings["total"] = sum(timings.values())
        return {
            "text": text,
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
            "analysis": analysis,
            "timings": timings,
        }


if __name__ == "__main__":  # python predict.py some_file.wav
    from audio_processing import load_audio

    result = SpeechEmotionPipeline().run(load_audio(sys.argv[1]))
    print("Transcript :", result["text"])
    print("Emotion    :", result["emotion"], f"({result['confidence']:.0%})")
    print("Analysis   :", result["analysis"])
    print("Timings (s):", {k: round(v, 2) for k, v in result["timings"].items()})
