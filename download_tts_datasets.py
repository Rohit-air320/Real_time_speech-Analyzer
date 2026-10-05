"""Downloads IndicTTS Hindi/Marathi samples for speech-to-text accuracy testing."""
from datasets import load_dataset

for name, repo in [("hindi", "SPRINGLab/IndicTTS-Hindi"), ("marathi", "SPRINGLab/IndicTTS_Marathi")]:
    print(f"Downloading {name} sample...")
    ds = load_dataset(repo, split="train[:50]")  # just the first 50 clips, enough to test
    ds.save_to_disk(f"./tts_data_{name}")
    print(f"{name}: saved {len(ds)} clips to ./tts_data_{name}") 