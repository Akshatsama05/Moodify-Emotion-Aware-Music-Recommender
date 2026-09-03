🎧 Moodify — Emotion-Aware Music Recommender

Describe how you feel. Moodify turns your text into an emotion-aware music recommendation experience.

Moodify is an NLP + machine learning music recommender built with Python, scikit-learn, and Streamlit. It analyzes a user's natural-language mood, predicts one of six emotions, and ranks songs from a multilingual Spotify metadata catalogue using emotion relevance, TF-IDF similarity, and popularity.

Emotions: sadness · joy · love · anger · fear · surprise

✨ Highlights

🧠 Six-class Linear SVM emotion classifier

🔤 TF-IDF unigrams + bigrams for text representation

🎵 Metadata/content-based six-emotion song relevance

🌍 English, Hindi, and Punjabi catalogue support

🔎 TF-IDF similarity between the user's mood text and song metadata

📈 Popularity as a secondary ranking signal

🎯 Conservative emotion-evidence gate for stronger candidates

🔀 Emotion-aware shuffle

🖼️ Album artwork + song metadata

🎧 Spotify links when available

🖥️ Streamlit application

☁️ Google Colab launch workflow

🧪 Offline project validation and smoke tests

🔬 Lightweight TF-IDF prediction explainability

🧩 Optional multimodal integration for externally supplied lyrics/audio features

👥 Optional local human-annotation workflow for future catalogue evaluation

🎯 Problem

A user may know exactly how they feel without knowing what music fits that feeling:

"I know how I feel, but I don't know what music fits that feeling."

Moodify explores whether natural-language mood descriptions can be converted into an emotion-aware recommendation experience without requiring listening history or collaborative-filtering data.

💡 How It Works

┌──────────────────────────┐
│ User mood description    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ TF-IDF text features     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Balanced Linear SVM      │
└────────────┬─────────────┘
             ↓
   predicted emotion
   + decision-score weights
             ↓
┌──────────────────────────┐
│ Catalogue relevance      │
│ metadata + content cues  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ TF-IDF query similarity  │
│ + popularity             │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Ranked Top 10 songs      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Streamlit UI + Spotify   │
└──────────────────────────┘

Recommendation score

Rank Score =
    0.70 × emotion relevance
  + 0.20 × TF-IDF similarity
  + 0.10 × normalized popularity

Emotion relevance is deliberately the dominant signal. Similarity personalizes results within the selected emotion, while popularity is a smaller secondary signal.

🤖 Machine Learning

Text representation

Moodify uses scikit-learn TfidfVectorizer with:

lowercase text

unigrams + bigrams

min_df=2

max_features=12000

sublinear_tf=True

Classifier

The production classifier is a balanced LinearSVC:

LinearSVC(
    class_weight="balanced",
    random_state=42
)

The six production classes are preserved directly:

sadness
joy
love
anger
fear
surprise

No three-emotion production mapping is used.

Model artifact

The trained production pipeline is stored at:

models/moodify_model.joblib

It contains the TF-IDF vectorizer and Linear SVM used by the application.

📊 Model Performance

The final model is fitted using the 16,000 training + 2,000 validation examples and evaluated on the held-out 2,000-example test split.

Metric

Score

Accuracy

89.05%

Balanced Accuracy

85.26%

Macro-F1

84.38%

Per-class results

Emotion

Precision

Recall

F1

Support

Sadness

93.51%

91.74%

92.62%

581

Joy

92.94%

90.94%

91.93%

695

Love

74.46%

86.16%

79.88%

159

Anger

86.27%

89.09%

87.66%

275

Fear

88.26%

83.93%

86.04%

224

Surprise

66.67%

69.70%

68.15%

66

Important: these metrics evaluate user-text emotion classification. They do not measure objective recognition of song emotion.

Confusion matrix



🎵 Recommendation Engine

The recommender separates user-text classification from catalogue song relevance.

1. Metadata/content relevance

The bundled catalogue contains metadata including:

song title

artist

album

genre

language

popularity

release information

artwork

Spotify URL

Mood-related numeric fields in the supplied raw catalogue are empty. The project therefore does not treat them as song-emotion ground truth.

Instead, Moodify derives six transparent relevance signals using:

emotion-specific lexical cues

phrase-level cues

English and Romanized Hindi/Punjabi vocabulary

TF-IDF similarity to emotion prototype descriptions

a small genre prior for selected cues

Artist names are excluded from emotion/content scoring to reduce artist-identity leakage.

2. Six-way emotion profile

Each processed song receives:

emotion_prob_sadness
emotion_prob_joy
emotion_prob_love
emotion_prob_anger
emotion_prob_fear
emotion_prob_surprise

These values form a comparable six-way relevance profile. A conservative display label can be unclassified when evidence is too weak.

3. Evidence gate

Moodify uses a 0.10 emotion-evidence threshold as a first-tier preference:

Rank candidates using the combined score.

Prefer tracks with emotion evidence ≥ 0.10.

If at least 10 strong candidates exist, return the strongest 10.

If fewer than 10 exist, use the next-best candidates only to complete the requested list.

This is a ranking safeguard, not human validation of song emotion.

4. Duplicate handling

The recommender removes duplicate song_name + artist combinations so repeated catalogue records do not unnecessarily consume recommendation slots.

🌍 Multilingual Catalogue

The bundled catalogue contains 2,878 songs:

Language

Songs

Hindi

1,000

Punjabi

979

English

899

Total

2,878

The application provides language filtering for:

All

English

Hindi

Punjabi

Language filtering changes the recommendation pool; it does not retrain the classifier.

🖥️ Streamlit App

The main application is app.py.

The UI includes:

natural-language mood input

example prompts

six-emotion selection

predicted emotion display

emotion score feedback

language filtering

Top 10 recommendations

emotion-aware shuffle

album artwork

song metadata

Spotify buttons

explanation of the NLP/ranking process

project notes and limitations

The application loads:

models/moodify_model.joblib
data/processed/labelled_catalogue.csv

An optional multimodal processed catalogue can be preferred when generated by the multimodal preparation workflow.

🧩 Optional Multimodal Extension

Moodify includes an integration layer in src/multimodal.py for legitimate externally supplied emotion profiles.

Metadata emotion profile
          │
          ├───────────────┐
          ▼               ▼
   Lyrics profile    Audio profile
          │               │
          └───────┬───────┘
                  ▼
            Fused profile
                  ▼
          Recommendation ranker

The bundled repository does not contain fabricated lyrics, audio features, or human labels.

Optional features must be joined using the exact catalogue id, not title alone.

See data/external/README.md for the expected schema and preparation command.

👥 Human Validation Workflow

The current catalogue has no supplied human song-emotion annotations, so the repository does not claim human-validated catalogue accuracy.

Instead, a reproducible annotation workflow is included:

reports/human_validation/annotation_queue_300.csv

The queue contains:

100 English songs

100 Hindi songs

100 Punjabi songs

300 songs total

Tools:

tools/annotation_app.py
tools/evaluate_human_validation.py

The evaluation workflow supports measures including:

accuracy

balanced accuracy

macro-F1

confusion matrix

annotator agreement / Fleiss' kappa

recommendation Precision@K

NDCG@K

artist diversity

These metrics are meaningful only after real annotations are supplied.

🏗️ Project Structure

Moodify-Emotion-Aware-Music-Recommender/
│
├── app.py                              # Streamlit application
├── run_local.py                        # Local launcher + health check
├── run_local.bat                       # Windows launcher
├── run_ngrok.bat                       # Optional ngrok launcher
├── validate_project.py                 # Offline integrity checks
├── requirements.txt                    # Dependencies
│
├── src/
│   ├── data.py                         # Catalogue loading/cleaning
│   ├── emotion_data.py                 # Six-emotion definitions
│   ├── labeling.py                     # Catalogue emotion relevance
│   ├── modeling.py                     # Model training/evaluation
│   ├── recommender.py                  # Prediction + ranking
│   └── multimodal.py                   # Optional feature fusion
│
├── models/
│   └── moodify_model.joblib            # Production model
│
├── data/
│   ├── raw/spotify_metadata_catalogue.csv
│   ├── processed/labelled_catalogue.csv
│   └── external/README.md
│
├── notebooks/
│   ├── Moodify_End_to_End.ipynb
│   ├── Moodify_End_to_End_executed.ipynb
│   └── Streamlit_Colab_Launcher.py
│
├── reports/
│   ├── figures/                         # Evaluation visualizations
│   ├── human_validation/               # Annotation workflow
│   ├── catalogue_emotion_recommendation_audit.csv
│   ├── catalogue_emotion_score_summary.csv
│   └── LIMITATION_STATUS.md
│
├── tools/
│   ├── annotation_app.py
│   ├── create_annotation_queue.py
│   ├── evaluate_human_validation.py
│   └── prepare_multimodal_features.py
│
├── emotion_train.csv                   # 16,000 training rows
├── emotion_validation.csv              # 2,000 validation rows
├── emotion_test.csv                    # 2,000 test rows
├── emotion_model_metrics.json
├── emotion_model_report.md
├── Moodify_Master_Colab.ipynb
├── LOCAL_RUN_GUIDE.md
├── DEPENDENCY_MAP.md
└── FINAL_FIX_REPORT.md

🛠️ Tech Stack

Category

Technologies

Language

Python

ML

scikit-learn, Linear SVM

NLP

TF-IDF, lexical/phrase cues

Data

Pandas, NumPy

Visualization

Matplotlib, Seaborn

Model persistence

Joblib

UI

Streamlit

Notebook

Jupyter Notebook, Google Colab

Optional tunneling

Cloudflare Quick Tunnel, ngrok

🚀 Run Locally

1. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Start Moodify

streamlit run app.py

Or use the project launcher:

python run_local.py local

The local launcher starts Streamlit on port 8501 and performs a health check.

For Windows, run_local.bat is also available.

Optional ngrok mode

If NGROK_AUTHTOKEN is configured:

python run_local.py ngrok

ngrok is not required for normal local use.

☁️ Google Colab

The repository includes:

Moodify_Master_Colab.ipynb

The end-to-end workflow covers model training/evaluation, catalogue processing, artifact verification, recommendation smoke tests, and Streamlit launching.

A dedicated launcher is also available:

notebooks/Streamlit_Colab_Launcher.py

It starts the real Streamlit application, checks its local health endpoint, and attempts a Cloudflare Quick Tunnel. An ngrok fallback is available when NGROK_AUTHTOKEN is supplied.

🧪 Validate the Project

Run:

python validate_project.py

The validation script checks the model classes, catalogue size, six emotion score/profile columns, recommendation paths, normalized model scores, validation queue, and limitation documentation.

Expected output:

MODEL_CLASSES_OK
CATALOGUE_ROWS_OK 2878
SIX_EMOTION_RECOMMENDATIONS_OK
SVM_SCORE_NORMALIZATION_OK
LIMITATION_STATUS_OK

🔬 Explainability

src/recommender.py includes explain_prediction().

It inspects the TF-IDF representation and returns influential terms for the predicted Linear SVM class, providing a lightweight view of which text features contributed to a prediction.

⚠️ Limitations & Responsible Interpretation

Moodify intentionally separates validated ML results from heuristic catalogue signals.

Validated

User-text emotion classification on a held-out 2,000-row test set.

Six-class production model.

Offline recommendation/integrity checks.

Deterministic metadata/content relevance pipeline.

Not currently validated

The bundled music catalogue does not provide:

human song-emotion labels;

lyrics;

audio-derived emotion features;

psychological ground truth for song emotion.

Therefore, Moodify should not claim that a song is objectively happy, sad, angry, etc. based on the current catalogue layer.

The supported interpretation is:

Moodify ranks songs using metadata/content-based emotion relevance signals.

Human annotation and multimodal integrations are provided as extension paths for future evidence-based evaluation.

🔮 Future Improvements

Human-labelled song-emotion dataset with multiple independent annotators.

Lyrics-based emotion modelling using legitimately available/licensed data.

Audio emotion modelling using validated audio features or models.

Probability calibration for the Linear SVM instead of treating softmax-transformed decision values as probability-like ranking weights.

Learning-to-rank using real human relevance judgements or user feedback.

Personalization using likes, skips, history, or playlist interactions.

Recommendation evaluation using Precision@K, NDCG@K, diversity, and coverage on genuinely labelled data.

🧱 Engineering Decisions

Why Linear SVM?

TF-IDF creates a high-dimensional sparse text representation where linear classifiers are efficient and effective. Linear SVM is therefore a practical choice for the six-class text classification task.

Why balanced class weights?

The emotion classes are not equally represented. class_weight="balanced" helps reduce bias toward the larger classes.

Why separate text classification from song relevance?

The text dataset has supervised emotion labels, while the bundled song catalogue does not. Separating the two prevents unsupported song-emotion ground-truth claims.

Why exclude artists from mood scoring?

Artist identity can introduce accidental correlations unrelated to the mood expressed by the metadata. Mood/content scoring therefore uses song title, album, genre, and language rather than artist identity.

Why use an evidence gate?

Metadata can be sparse. The evidence gate gives stronger candidates first priority while still allowing the ranking system to complete a Top 10 list when enough catalogue candidates exist.

📌 Project Status

Functional ML/NLP prototype with:

trained six-class emotion model;

multilingual 2,878-song catalogue;

metadata/content recommendation layer;

Streamlit interface;

local and Colab launch workflows;

offline validation;

optional human-validation tooling;

optional multimodal integration path.

The strongest supported project claim is:

Six-class emotion classification of user text combined with metadata/content-based music ranking.

👨‍💻 Author

Akshat Sajwan
B.Tech Computer Science student focused on Python, Machine Learning, NLP, and AI/ML application development.

📚 Documentation

For deeper technical details, see:

emotion_model_report.md — model evaluation

FINAL_FIX_REPORT.md — engineering fixes and limitations

DEPENDENCY_MAP.md — component dependencies

LOCAL_RUN_GUIDE.md — local execution notes

reports/LIMITATION_STATUS.md — evidence/limitation status

data/external/README.md — optional multimodal schema

reports/human_validation/ANNOTATION_INSTRUCTIONS.md — annotation workflow

⭐ Project Summary

Moodify = NLP + Machine Learning + Recommendation + Streamlit

The project demonstrates an end-to-end pipeline from natural-language input → emotion classification → content-aware relevance → ranking → interactive music recommendations, while keeping measured model performance separate from unvalidated song-emotion assumptions.
