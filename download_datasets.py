"""Downloads extra datasets for the generalization study. Run once: python download_datasets.py"""
import kagglehub

datasets = {
    "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",
    "savee": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
    "tess": "ejlok1/toronto-emotional-speech-set-tess",
    "besd": "pranuthi19/bilingual-emotion-speech-dataset-besd",
    "hindi_marathi": "swastikatinkhede/multilingual-speech-dataset",
    "gujarati": "yp271289/multi-language-emotion-dataset",
}

for name, slug in datasets.items():
    try:
        path = kagglehub.dataset_download(slug)
        print(f"{name}: {path}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")