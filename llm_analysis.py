"""LLM analysis: explains the transcript + detected emotion in simple words."""
import os

import config

PROMPT = """Transcript of what the person said: "{text}"
Emotion detected from their voice: {emotion} (confidence {confidence:.0%})
All emotion scores: {scores}

Write exactly 3 short sentences:
1. What the speaker seems to be feeling.
2. Whether the words and the tone of voice agree.
3. A one-line summary of the message.
   Do not invent details that are not in the transcript.
   Always reply in English, even if the transcript is in another language."""


class LLMAnalyzer:
    def __init__(self):
        self.backend = config.LLM_BACKEND
        if self.backend == "openai":
            from openai import OpenAI

            self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_LLM)
            self.model = AutoModelForCausalLM.from_pretrained(config.LOCAL_LLM)
            self.model.eval()

    def analyze(self, text, emotion, confidence, scores) -> str:
        score_text = ", ".join(f"{k} {v:.0%}" for k, v in scores.items())
        prompt = PROMPT.format(text=text, emotion=emotion, confidence=confidence, scores=score_text)
        messages = [
            {"role": "system", "content": "You are a clear and concise assistant."},
            {"role": "user", "content": prompt},
        ]
        if self.backend == "openai":
            reply = self.client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=messages, max_tokens=150, temperature=0.3
            )
            return reply.choices[0].message.content.strip()

        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":  # quick test with made-up values
    demo_scores = {"happy": 0.7, "sad": 0.05, "angry": 0.05, "neutral": 0.1, "fear": 0.05, "surprise": 0.05}
    print(LLMAnalyzer().analyze("I got the job today!", "happy", 0.7, demo_scores))
