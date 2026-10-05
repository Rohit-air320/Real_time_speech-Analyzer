<<<<<<< HEAD
# Real-Time Speech Emotion Analysis

Microphone -> Whisper (speech-to-text) -> Wav2Vec2 (emotion) -> LLM (analysis) -> Streamlit dashboard.
Emotions: happy, sad, angry, neutral, fear, surprise.

## 1. Setup (Python 3.10 or 3.11)

Windows:
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Mac / Linux:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## 2. Test each module (first run downloads the models, needs internet)

Use any short English .wav file (for example one RAVDESS clip):

    python speech_to_text.py my_audio.wav     # prints the transcript
    python emotion_model.py my_audio.wav      # prints the emotion + scores
    python llm_analysis.py                    # prints a sample analysis
    python predict.py my_audio.wav            # full pipeline, with timings

## 3. Run the demo

    streamlit run app.py

Allow the microphone in the browser, click the mic, speak, click stop.

## 4. Baseline + results for the report (needs the RAVDESS dataset)

Download "Audio_Speech_Actors_01-24.zip" from https://zenodo.org/records/1188976 and unzip it
inside `dataset/` (you will get dataset/Actor_01 ... dataset/Actor_24). Then:

    python train.py         # trains the SVM baseline, saves models/svm_baseline.joblib
    python evaluation.py    # saves results/metrics.json and the graphs

## Settings

Everything you may want to change (model sizes, LLM backend) is in `config.py`.
To use OpenAI instead of the free local LLM: `pip install openai`, set `LLM_BACKEND = "openai"`
and set the OPENAI_API_KEY environment variable.

## Important note for the report

The pretrained Wav2Vec2 model was fine-tuned on RAVDESS, SAVEE and TESS, so it has probably already
seen the RAVDESS test actors. Its accuracy on RAVDESS is therefore optimistic and not a fair
comparison with the SVM, which only trains on actors 1-16. State this in the report.
=======
# Real_time_speech-Analyzer
>>>>>>> a884f1a428f98bcdfd809cd7b18d0d92f739b5dc
