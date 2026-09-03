# 🎧 Moodify — Emotion-Aware Music Recommender

> **Describe how you feel. Moodify finds music that fits the mood.**

Moodify is an end-to-end **NLP and machine-learning music recommendation system** that converts a user's natural-language feeling into a six-emotion playlist direction and ranks relevant songs from a multilingual catalogue.

The project combines **TF-IDF text representation, a class-balanced Linear SVM emotion classifier, metadata/content-based song-emotion relevance, TF-IDF query similarity, popularity ranking, multilingual cues, and a Streamlit interface**.

> **Evidence boundary:** The bundled song catalogue contains metadata, but does not contain human-validated song-emotion labels, lyrics, or audio-derived emotion features. Catalogue emotion values are therefore treated as **metadata/content relevance signals**, not ground-truth claims about how a song objectively feels.

---

## 📌 Project Overview

Moodify addresses a simple user problem:

> **"I know how I feel, but I don't know what music fits that feeling."**

Instead of requiring the user to select a predefined genre or playlist, Moodify allows them to describe their current feeling in ordinary language.

For example:

> *"I had a rough day and just want something quiet and emotional."*

The system interprets the text, predicts the dominant emotion, and ranks relevant songs from the catalogue.

### Production Emotion Classes

| Emotion | Meaning |
|---|---|
| 😢 `sadness` | Sadness, grief, loneliness, heartbreak |
| 😊 `joy` | Happiness, celebration, excitement |
| ❤️ `love` | Romance, affection, attachment |
| 😠 `anger` | Anger, frustration, hostility, conflict |
| 😨 `fear` | Fear, anxiety, danger, uncertainty |
| 😲 `surprise` | Surprise, shock, amazement, unexpected events |

The production path consistently uses these six classes.

---

## 🎯 Problem Statement

Emotion-aware recommendation involves two different technical problems.

### 1. Understand the user's text

Users may describe emotions indirectly rather than using a single emotion keyword.

For example:

> *"I had a rough day and just want something quiet and emotional."*

The system needs to infer the dominant emotional direction from natural language.

### 2. Find songs that fit the emotion

The bundled catalogue does not provide reliable human-labelled song-emotion ground truth.

Therefore, Moodify separates:

**User-text emotion classification**

from:

**Catalogue emotion relevance**

rather than presenting metadata heuristics as scientifically validated song-emotion labels.

---

## 💡 Solution

Moodify follows this workflow:

**User Mood Description → TF-IDF Vectorization → Balanced Linear SVM → Predicted Emotion → Catalogue Emotion Relevance → Query Similarity → Popularity → Evidence-Aware Ranking → Top-10 Recommendations**

The architecture is intentionally modular so that future human-labelled, lyrics-based, or audio-based emotion models can be integrated without replacing the core recommendation system.

---

## ✨ Key Capabilities

- 🧠 Six-class natural-language emotion classification
- 📚 TF-IDF unigram and bigram representation
- ⚖️ Class-balanced Linear SVM
- 📊 Held-out model evaluation
- 🎵 2,878-song catalogue
- 🌍 English, Hindi, and Punjabi coverage
- 🔤 English and Romanized Hindi/Punjabi emotion cues
- 🧩 Phrase-level emotion evidence
- 🧮 Six-way catalogue emotion relevance profiles
- 🔎 TF-IDF query-to-song similarity
- 📈 Popularity as a secondary ranking signal
- 🛡️ Conservative emotion-evidence gate
- 🔁 Duplicate title/artist handling
- 🔀 Mood-constrained shuffle
- 🖼️ Album artwork
- 🎧 Spotify links
- 🖥️ Streamlit application
- ☁️ Google Colab workflow
- 🔬 Human-validation workflow
- 🧩 Optional lyrics/audio multimodal integration
- ✅ Project regression and health checks
- 🔍 Explicit limitation and evidence documentation

---

## 🏗️ System Architecture

### High-Level Flow

**User → Natural-Language Mood → TF-IDF → Linear SVM → Emotion Prediction → Catalogue Relevance → Query Similarity → Popularity → Ranking → Top-10 Recommendations → Streamlit**

### Catalogue Relevance Layer

Moodify uses available metadata such as:

- song title
- album
- genre
- language

The relevance system combines:

- lexical emotion cues
- phrase-level cues
- TF-IDF emotion prototypes
- cosine similarity
- weak genre cues
- multilingual vocabulary

Artist names are excluded from the primary mood-scoring text to reduce artist-identity leakage.

---

# 🧠 Emotion Classification

## TF-IDF Representation

The production classifier uses TF-IDF with:

| Parameter | Value |
|---|---|
| Lowercase | `True` |
| N-gram range | `(1, 2)` |
| Minimum document frequency | `2` |
| Maximum features | `12,000` |
| Sublinear TF | `True` |

This allows the model to learn from both individual words and two-word phrases.

Examples include:

- `lonely`
- `heartbreak`
- `very happy`
- `broken heart`

---

## 🤖 Model Selection

The modelling workflow compares:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

The production model uses:

**TF-IDF Vectorizer → LinearSVC**

with balanced class weighting.

The trained production artifact is:

`models/moodify_model.joblib`

---

# 📊 Dataset

## Supervised Emotion Dataset

The project uses a fixed six-class dataset containing:

| Split | Rows |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Held-out Test | 2,000 |
| **Total** | **20,000** |

The held-out test set remains separate for final evaluation.

### Test Set Distribution

| Emotion | Support |
|---|---:|
| `sadness` | 581 |
| `joy` | 695 |
| `love` | 159 |
| `anger` | 275 |
| `fear` | 224 |
| `surprise` | 66 |
| **Total** | **2,000** |

The class imbalance is one reason the final classifier uses balanced class weighting.

---

# 🎵 Music Catalogue

The bundled catalogue contains:

### **2,878 Tracks**

| Language | Tracks |
|---|---:|
| 🇬🇧 English | 899 |
| 🇮🇳 Hindi | 1,000 |
| 🇮🇳 Punjabi | 979 |
| **Total** | **2,878** |

The raw catalogue includes fields such as:

- `id`
- `song_name`
- `singer`
- `album`
- `release_date`
- `cover_image`
- `spotify_url`
- `popularity`
- `genre`
- `language`
- `language_evidence`
- `source`

The preprocessing layer also creates fields such as:

- `artist_text`
- `release_year`
- `text`
- `recommendation_text`

### Recommendation Text

The recommendation representation focuses on:

**song name + album + genre + language**

Artist information is retained for display but excluded from the primary recommendation text used for mood similarity.

---

# 📈 Model Evaluation

The final Linear SVM is evaluated on the held-out 2,000-example test set.

## Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | **89.05%** |
| Balanced Accuracy | **85.26%** |
| Macro-F1 | **84.38%** |

## Per-Class Performance

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Sadness | 93.51% | 91.74% | 92.62% |
| Joy | 92.94% | 90.94% | 91.93% |
| Love | 74.46% | 86.16% | 79.88% |
| Anger | 86.27% | 89.09% | 87.66% |
| Fear | 88.26% | 83.93% | 86.04% |
| Surprise | 66.67% | 69.70% | 68.15% |

### Interpretation

The overall accuracy is strong, but the macro-F1 and per-class metrics are important because the emotion classes are imbalanced.

- `sadness` and `joy` are the strongest classes.
- `love`, `anger`, and `fear` show useful but imperfect separation.
- `surprise` is the weakest class and also has the smallest test support.

> **Important:** These metrics evaluate **emotion classification from user text**. They do not measure objective song-emotion accuracy.

---

# 🎵 Catalogue Emotion Profiling

The catalogue-side emotion layer creates six-way relevance values for tracks.

The processed catalogue can contain:

- `emotion_sadness`
- `emotion_joy`
- `emotion_love`
- `emotion_anger`
- `emotion_fear`
- `emotion_surprise`

and normalized profile values:

- `emotion_prob_sadness`
- `emotion_prob_joy`
- `emotion_prob_love`
- `emotion_prob_anger`
- `emotion_prob_fear`
- `emotion_prob_surprise`

These values represent **metadata/content relevance**, not human-labelled probabilities.

---

## 🔤 Lexical Evidence

Emotion-specific vocabularies include English and multilingual/Romanized cues.

For example, the `love` vocabulary contains terms such as:

`love`, `romance`, `affection`, `relationship`, `ishq`, `pyaar`, `pyar`, `mohabbat`, `prem`, `sanam`, `mahi`, `yaara`

The system also supports phrase-level cues because multi-word expressions can provide stronger evidence than isolated tokens.

---

## 🔎 Emotion Prototype Similarity

Moodify creates six emotion prototypes and calculates TF-IDF cosine similarity between catalogue metadata and those prototypes.

This provides an additional relevance signal beyond direct keyword matching.

The prototype vocabulary also includes multilingual cues so that Hindi and Punjabi metadata is not evaluated only against English emotion terms.

---

## 🎼 Weak Genre Priors

Genre information can provide a small supporting signal for selected emotions.

This contribution is deliberately limited because:

> **Genre is not reliable ground truth for emotional content.**

---

# 🧮 Recommendation Engine

The recommender does not simply filter songs using a single emotion label.

Instead, it combines multiple ranking signals.

### Ranking Formula

**Rank Score = 0.70 × Emotion Relevance + 0.20 × TF-IDF Similarity + 0.10 × Popularity**

| Signal | Weight | Purpose |
|---|---:|---|
| Emotion relevance | **70%** | Primary mood-matching signal |
| TF-IDF similarity | **20%** | Matches the user's actual wording |
| Popularity | **10%** | Secondary discovery signal |

The intended priority is:

**Emotion Fit → Query Relevance → Popularity**

This keeps the system emotion-first rather than popularity-first.

---

# 🛡️ Evidence Gate

Moodify uses:

`MIN_EMOTION_EVIDENCE = 0.10`

Strong-evidence candidates are prioritized first.

If at least ten strong candidates are available, the Top-10 list is selected from those candidates.

If fewer than ten strong candidates are available, the system can use the next-best candidates to complete the recommendation list.

This provides a practical balance between:

- recommendation precision
- catalogue coverage

The evidence gate is a ranking safeguard. It does not create ground-truth song-emotion labels.

---

# 🔁 Duplicate Handling

Recommendations remove duplicate combinations of:

**song title + artist**

This prevents repeated catalogue records from consuming multiple recommendation slots.

Different artists with the same song title can still remain separate recommendations.

---

# 🔀 Shuffle Mode

Shuffle mode does not sample randomly from the entire catalogue.

Instead:

**Selected Emotion → Recommendation Candidate Set → Shuffle**

This keeps shuffled recommendations within the selected mood direction.

---

# 🔎 Query Similarity

The recommendation layer uses TF-IDF cosine similarity between:

**User Mood Description ↔ Song Recommendation Text**

This allows the actual wording of the user's request to influence the ordering.

For example:

> *"I feel lonely and nostalgic."*

can favour catalogue metadata that is textually closer to those concepts.

Query similarity remains secondary to emotion relevance.

---

# 📈 Popularity Signal

Popularity contributes only **10%** to the final ranking.

It is normalized and used as a secondary signal rather than allowing the most popular songs to dominate the recommendations.

---

# 🌍 Multilingual Support

Moodify's catalogue covers:

- English
- Hindi
- Punjabi

The metadata relevance layer also supports Romanized Indian-language cues.

Examples include terms related to:

- sadness
- joy
- love
- anger
- fear
- surprise

The Streamlit application provides:

**All · English · Hindi · Punjabi**

The language selector changes the recommendation pool but does not retrain the emotion classifier.

---

# 🖥️ Streamlit Application

The production UI is implemented in:

`app.py`

### User Flow

**Describe Feeling → Analyze Mood → Select Language → Generate Recommendations → Browse Top 10 → Open Song on Spotify**

### Interface Features

- Natural-language mood input
- Example prompts
- Emotion analysis
- Six-emotion selection
- Language filtering
- Top-10 recommendation cards
- Album artwork
- Artist information
- Album information
- Popularity
- Spotify links
- Shuffle mode
- Recommendation/model information
- Limitation information

---

# 🔬 Explainability

Moodify includes an `explain_prediction()` function in:

`src/recommender.py`

For a Linear SVM prediction, the system can inspect TF-IDF vocabulary features and classifier coefficients to identify terms associated with a predicted emotion.

This provides lightweight model-level explanation without claiming complete model interpretability.

---

# 🧩 Optional Multimodal Architecture

The repository contains an optional multimodal integration layer:

`src/multimodal.py`

It is designed to accept externally generated six-way emotion features from:

- lyrics
- audio

Optional features are joined using the stable catalogue:

`id`

Title-only matching is avoided because titles can collide across different artists, remixes, and versions.

### Fusion Weights

The intended default weighting is:

| Modality | Weight |
|---|---:|
| Metadata | 45% |
| Lyrics | 30% |
| Audio | 25% |

Missing modalities can be excluded and the remaining weights renormalized.

> The current repository does **not** claim to contain validated lyrics/audio emotion data. This is an extension/integration layer.

---

# 👥 Human Validation Framework

The repository includes a reproducible local human-validation workflow.

### Annotation Queue

`reports/human_validation/annotation_queue_300.csv`

The queue contains:

| Language | Songs |
|---|---:|
| English | 100 |
| Hindi | 100 |
| Punjabi | 100 |
| **Total** | **300** |

### Annotation Interface

`tools/annotation_app.py`

### Evaluation

`tools/evaluate_human_validation.py`

Future evaluation can include:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Confusion Matrix
- Majority Agreement
- Fleiss' Kappa
- Precision@5
- Precision@10
- NDCG@5
- NDCG@10
- Artist Diversity

> **Current status:** The repository is human-validation ready, but human song-emotion validation has not yet been completed.

No human-validation metrics are fabricated.

---

# 🗂️ Project Structure

### Application

- `app.py` — Streamlit application
- `run_local.py` — local launcher
- `run_local.bat` — Windows launcher
- `run_ngrok.bat` — optional ngrok launcher

### Machine Learning

- `src/modeling.py` — model construction, training, comparison, and evaluation
- `src/emotion_data.py` — canonical six-emotion definitions
- `models/moodify_model.joblib` — trained production model

### Data

- `src/data.py` — catalogue loading, validation, and preprocessing
- `data/raw/spotify_metadata_catalogue.csv` — raw catalogue
- `data/processed/labelled_catalogue.csv` — processed catalogue
- `data/external/` — optional external-feature documentation

### Recommendation

- `src/labeling.py` — metadata/content emotion relevance
- `src/recommender.py` — prediction and ranking logic
- `src/multimodal.py` — optional multimodal integration

### Tools

- `tools/annotation_app.py` — human annotation interface
- `tools/create_annotation_queue.py` — validation queue generation
- `tools/evaluate_human_validation.py` — human-validation evaluation
- `tools/prepare_multimodal_features.py` — optional feature preparation

### Notebooks

- `Moodify_Master_Colab.ipynb`
- `Moodify_Master_Colab_executed.ipynb`
- `notebooks/Moodify_End_to_End.ipynb`
- `notebooks/Moodify_End_to_End_executed.ipynb`
- `notebooks/Streamlit_Colab_Launcher.py`

### Data and Evaluation Files

- `emotion_train.csv`
- `emotion_validation.csv`
- `emotion_test.csv`
- `emotion_model_metrics.json`
- `emotion_model_report.md`
- `reports/`
- `validation_summary.md`
- `final_quality_audit.md`
- `integration_verification.md`
- `FINAL_FIX_REPORT.md`
- `DEPENDENCY_MAP.md`
- `LOCAL_RUN_GUIDE.md`

---

# 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn, Linear SVM |
| NLP | TF-IDF, cosine similarity |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Application | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook, Google Colab |

---

# 🚀 Installation

## Requirements

Recommended:

- Python 3.10+
- pip
- Git
- Modern web browser

### Install dependencies

Run:

`pip install -r requirements.txt`

---

# ▶️ Run Locally

### Direct Streamlit launch

Run:

`streamlit run app.py`

Then open:

`http://127.0.0.1:8501`

### Project launcher

Run:

`python run_local.py local`

### Windows

You can also run:

`run_local.bat`

---

# 🌐 Optional ngrok Mode

For a temporary public URL, use:

`python run_local.py ngrok`

The environment variable:

`NGROK_AUTHTOKEN`

must be configured.

The token should remain outside the repository and should never be committed.

---

# ☁️ Google Colab

The repository includes:

`Moodify_Master_Colab.ipynb`

and:

`Moodify_Master_Colab_executed.ipynb`

The master notebook covers:

1. Dataset preparation
2. Model training
3. Model comparison
4. Model evaluation
5. Catalogue processing
6. Emotion relevance generation
7. Recommendation testing
8. Model artifact verification
9. Streamlit launch
10. Health checks

---

# 🧪 Validation & Regression Checks

Run:

`python validate_project.py`

The project validation checks important assumptions such as:

- production model availability
- six emotion classes
- catalogue row count
- six-way emotion features
- recommendation generation
- SVM score normalization
- documented limitation status

The project has verified checks covering:

- `MODEL_CLASSES_OK`
- `CATALOGUE_ROWS_OK`
- `SIX_EMOTION_RECOMMENDATIONS_OK`
- `SVM_SCORE_NORMALIZATION_OK`
- `LIMITATION_STATUS_OK`

---

# 📋 Reproducibility

The repository contains:

- trained model artifact
- training data
- validation data
- held-out test data
- processed catalogue
- evaluation metrics
- notebooks
- validation scripts
- project documentation

The production model artifact is:

`models/moodify_model.joblib`

This allows the main supervised workflow to be inspected without retraining the model from scratch.

---

# ⚠️ Limitations

## Song Emotion Is Not Ground Truth

The bundled music catalogue does not provide validated human song-emotion labels.

It also does not contain bundled lyrics or validated audio-derived emotion predictions.

Therefore:

**User text → Emotion classification**

is a supervised ML task with held-out evaluation.

Whereas:

**Song metadata → Emotion relevance**

is a metadata/content-based recommendation signal.

### What this means

Moodify should not be interpreted as:

- a psychological assessment tool
- a clinical emotion detector
- a scientifically validated song-emotion classifier
- an objective detector of the emotional state of a song

The reported **89.05% accuracy** specifically refers to **user-text emotion classification**.

---

# 🧭 Responsible Interpretation

Moodify should be described as:

> **An emotion-aware music recommendation system using supervised user-text emotion classification and metadata/content-based song relevance.**

The project intentionally separates:

**Model Prediction → Catalogue Relevance → Human/Multimodal Ground Truth**

This makes the current implementation more transparent and provides a clear path for future validation.

---

# 🔮 Future Improvements

### Human-Validated Song Emotion Dataset

Collect independent human annotations and evaluate:

- agreement
- macro-F1
- balanced accuracy
- Precision@K
- NDCG@K
- artist diversity

### Lyrics-Based Emotion Modelling

Integrate validated lyrics-derived emotion features.

### Audio-Based Emotion Modelling

Add validated audio-derived emotion representations using acoustic features or learned audio embeddings.

### Multimodal Fusion

Combine:

**Metadata + Lyrics + Audio**

into a richer emotion representation.

### Better Multilingual NLP

Move beyond keyword/prototype cues toward stronger multilingual semantic models.

### Calibrated Uncertainty

Evaluate probability calibration and uncertainty estimation for the user-text classifier.

### Personalization

Use:

- likes
- skips
- saves
- replay behaviour
- listening history

to improve individual recommendations.

### Learning-to-Rank

With enough real preference data, replace manually weighted ranking with a learned recommendation model.

---

# 🧪 Technical Highlights

## Machine Learning

- Multi-class text classification
- TF-IDF feature engineering
- Linear SVM
- Class balancing
- Model comparison
- Held-out evaluation
- Confusion-matrix analysis

## NLP

- Unigrams and bigrams
- Phrase-level emotion cues
- Multilingual/Romanized vocabulary
- TF-IDF cosine similarity
- Emotion prototypes

## Recommendation Systems

- Content-based ranking
- Emotion-conditioned retrieval
- Weighted ranking
- Popularity normalization
- Evidence gating
- Duplicate suppression
- Constrained shuffle

## Software Engineering

- Modular `src/` architecture
- Reusable model/recommender functions
- Persisted model artifact
- Streamlit frontend
- Local launcher
- Google Colab workflow
- Validation scripts
- Human annotation tooling
- Optional multimodal integration

## Responsible ML

- No fabricated human labels
- No fabricated lyrics/audio features
- Explicit evidence boundaries
- Conservative relevance handling
- ID-based multimodal joins
- No claim that SVM scores are calibrated probabilities

---

# 📊 Project Snapshot

| Component | Status |
|---|---|
| Six-class NLP classifier | ✅ Implemented |
| TF-IDF + Linear SVM | ✅ Implemented |
| Held-out evaluation | ✅ Completed |
| Test Accuracy | **89.05%** |
| Balanced Accuracy | **85.26%** |
| Macro-F1 | **84.38%** |
| Music Catalogue | **2,878 tracks** |
| Languages | **English, Hindi, Punjabi** |
| Emotion-aware ranking | ✅ Implemented |
| Query similarity | ✅ Implemented |
| Popularity ranking | ✅ Implemented |
| Evidence-aware recommendation | ✅ Implemented |
| Streamlit application | ✅ Implemented |
| Local launcher | ✅ Included |
| Google Colab workflow | ✅ Included |
| Human-validation framework | ✅ Included |
| Human song-emotion validation | ⏳ Pending |
| Lyrics/audio multimodal data | ⏳ External data required |

---

# 🛠️ Example

A user enters:

> *"I had a really difficult day and feel lonely. I want something emotional."*

Moodify processes the request through:

**User Input → TF-IDF → Linear SVM → Emotion Prediction → Catalogue Relevance → Query Similarity → Popularity → Ranking → Top-10 Recommendations**

The user can then:

- choose a language
- inspect the mood direction
- shuffle recommendations
- view song metadata
- open songs on Spotify

---

# 📁 Important Reports

The repository also contains supporting engineering documentation:

| File | Purpose |
|---|---|
| `emotion_model_report.md` | Final supervised emotion-model evaluation |
| `emotion_model_metrics.json` | Machine-readable model metrics |
| `FINAL_FIX_REPORT.md` | Engineering corrections |
| `final_quality_audit.md` | Final quality audit |
| `integration_verification.md` | Integration verification |
| `validation_summary.md` | Validation summary |
| `DEPENDENCY_MAP.md` | Dependency documentation |
| `LOCAL_RUN_GUIDE.md` | Local execution guide |
| `vad_mapping_experiment.md` | VAD experiment documentation |
| `reports/LIMITATION_STATUS.md` | Evidence/limitation status |

---

# 🔐 Data & Evidence Policy

Moodify follows a simple principle:

> **If the repository does not contain the evidence, the project does not claim that the evidence exists.**

Therefore:

- Human song-emotion labels are not fabricated.
- Lyrics are not fabricated.
- Audio emotion features are not fabricated.
- Human-validation metrics are not fabricated.
- Metadata relevance is explicitly described as heuristic.
- SVM decision scores are not presented as calibrated probabilities.

---

# 📌 Final Takeaway

Moodify demonstrates a complete machine-learning application workflow:

**Data → Preprocessing → Feature Engineering → TF-IDF → Model Training → Model Evaluation → Emotion-Aware Ranking → Recommendation → Streamlit Application**

The project combines:

- Natural Language Processing
- Machine Learning
- Recommendation Systems
- Multilingual metadata processing
- Ranking algorithms
- Streamlit application development
- Validation tooling
- Responsible ML practices

The core design principle is the distinction between:

**What the model has actually learned → What the catalogue evidence supports → What still requires human or multimodal validation**

This keeps the current implementation technically transparent while providing a clear path toward a stronger, human-validated and multimodal emotion-aware recommendation system.

---

# 👨‍💻 Author

## Akshat Sajwan

**B.Tech Computer Science Student**

Interested in:

**Machine Learning · NLP · AI · Computer Vision · Python · Data Science**
