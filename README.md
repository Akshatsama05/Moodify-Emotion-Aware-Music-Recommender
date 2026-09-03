# 🎧 Moodify — Emotion-Aware Music Recommender

> **Describe how you feel. Moodify finds music that fits the mood.**

Moodify is an end-to-end **NLP + Machine Learning music recommendation system** that understands a user's natural-language mood and recommends relevant songs from a multilingual music catalogue.

The project combines **TF-IDF, a class-balanced Linear SVM, metadata-based emotion relevance, cosine similarity, popularity-aware ranking, multilingual filtering, and Streamlit** into an interactive recommendation application.

---

## ✨ Features

- 🧠 **Six-emotion NLP classification**
- 🎯 Emotion-aware music recommendations
- 📚 **TF-IDF + Linear SVM**
- 🌍 English, Hindi & Punjabi catalogue
- 🔎 Natural-language query similarity
- 🎵 Top-10 recommendations
- 📈 Popularity-aware ranking
- 🔀 Mood-aware shuffle
- 🖼️ Album artwork
- 🎧 Spotify links
- 🖥️ Streamlit web application
- ☁️ Google Colab workflow
- 🧪 Automated project validation
- 👥 Human-validation framework for future evaluation

---

## 🎯 Problem

Most simple music recommenders rely on listening history, predefined playlists, genres, or popularity.

But sometimes a user simply wants to say:

> *"I had a terrible day and want something emotional."*

Moodify explores a different approach:

```text
Natural-language mood
        ↓
Emotion classification
        ↓
Emotion-aware ranking
        ↓
Personalized recommendations
💡 How Moodify Works
              USER
                │
                ▼
       Natural-language mood
                │
                ▼
         TF-IDF Vectorization
                │
                ▼
       Balanced Linear SVM
                │
                ▼
        Predicted Emotion
                │
                ▼
     Catalogue Emotion Relevance
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     Lexical  TF-IDF   Genre
     Cues     Similarity  Cues
        │       │        │
        └───────┼────────┘
                ▼
       Query Similarity
                │
                ▼
          Popularity
                │
                ▼
        Ranking & Filtering
                │
                ▼
       Top-10 Recommendations
                │
                ▼
        Streamlit Application
🧠 Emotion Classification

Moodify uses six emotion classes:

Emotion	Description
😢 Sadness	Sadness, loneliness, grief, heartbreak
😊 Joy	Happiness, celebration, excitement
❤️ Love	Romance, affection, attachment
😠 Anger	Anger, frustration, conflict
😨 Fear	Fear, anxiety, uncertainty
😲 Surprise	Shock, amazement, unexpected events

The production classifier uses:

TF-IDF Vectorizer
        ↓
Linear SVM

with balanced class weighting to reduce the effect of class imbalance.

📊 Model Performance

The final model was evaluated on a 2,000-example held-out test set.

Metric	Score
Accuracy	89.05%
Balanced Accuracy	85.26%
Macro-F1	84.38%
Per-class F1
Emotion	F1
Sadness	92.62%
Joy	91.93%
Anger	87.66%
Fear	86.04%
Love	79.88%
Surprise	68.15%

Note: These metrics evaluate emotion classification from user text. They are not song-emotion accuracy metrics.

🎵 Recommendation Engine

Moodify combines multiple signals when ranking songs.

Ranking Formula
Rank Score =
    0.70 × Emotion Relevance
  + 0.20 × TF-IDF Similarity
  + 0.10 × Popularity
Why?
70% Emotion Relevance → keeps recommendations aligned with the user's mood
20% Query Similarity → matches the user's actual wording
10% Popularity → provides a secondary discovery signal

The goal is:

Emotion Fit
     ↓
Query Relevance
     ↓
Popularity

rather than simply recommending the most popular songs.

🎯 Example

A user enters:

I feel lonely and nostalgic. I want something emotional.

Moodify processes the request as:

User Text
   ↓
TF-IDF
   ↓
Linear SVM
   ↓
Emotion Prediction
   ↓
Catalogue Relevance
   ↓
Query Similarity
   ↓
Popularity
   ↓
Ranking
   ↓
Top 10 Songs

The user can then filter by language, shuffle the recommendations, and open songs through Spotify.

🌍 Multilingual Catalogue

The current catalogue contains 2,878 tracks:

Language	Tracks
🇬🇧 English	899
🇮🇳 Hindi	1,000
🇮🇳 Punjabi	979
Total	2,878

The recommendation layer also includes Romanized Indian-language emotion cues.

🖥️ Streamlit Application

The main application is implemented in:

app.py

The interface provides:

Natural-language mood input
Emotion analysis
Six-emotion selection
Language filtering
Top-10 recommendations
Album artwork
Artist and album information
Popularity
Spotify links
Shuffle mode
Recommendation/model information
🛠️ Tech Stack
Languages & Frameworks
Python
Streamlit
Machine Learning
Scikit-learn
Linear SVM
TF-IDF
Cosine Similarity
Data
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Model & Development
Joblib
Jupyter Notebook
Google Colab
📁 Project Structure
Moodify-Emotion-Aware-Music-Recommender/
│
├── app.py
├── requirements.txt
├── validate_project.py
├── run_local.py
├── run_local.bat
├── run_ngrok.bat
│
├── data/
│   ├── raw/
│   │   ├── spotify_metadata_catalogue.csv
│   │   └── ...
│   ├── processed/
│   │   └── labelled_catalogue.csv
│   └── external/
│
├── models/
│   └── moodify_model.joblib
│
├── src/
│   ├── data.py
│   ├── emotion_data.py
│   ├── labeling.py
│   ├── modeling.py
│   ├── multimodal.py
│   └── recommender.py
│
├── notebooks/
│   ├── Moodify_End_to_End.ipynb
│   ├── Moodify_End_to_End_executed.ipynb
│   └── Streamlit_Colab_Launcher.py
│
├── tools/
│   ├── annotation_app.py
│   ├── create_annotation_queue.py
│   ├── evaluate_human_validation.py
│   └── prepare_multimodal_features.py
│
├── reports/
│   ├── figures/
│   └── human_validation/
│
├── emotion_train.csv
├── emotion_validation.csv
├── emotion_test.csv
├── emotion_model_metrics.json
└── emotion_model_report.md
🚀 Getting Started
1. Clone the repository
git clone https://github.com/Akshatsama05/Moodify-Emotion-Aware-Music-Recommender.git
cd Moodify-Emotion-Aware-Music-Recommender
2. Install dependencies
pip install -r requirements.txt
3. Run Moodify
streamlit run app.py

Then open:

http://127.0.0.1:8501
🪟 Windows

You can also use:

run_local.bat

or:

python run_local.py local
☁️ Google Colab

The repository includes an end-to-end Colab workflow:

Moodify_Master_Colab.ipynb

The notebook covers:

Dataset preparation
Model training
Model evaluation
Catalogue processing
Recommendation testing
Model artifact verification
Streamlit launch

An executed notebook is also included:

Moodify_Master_Colab_executed.ipynb
🧪 Validation

Run:

python validate_project.py

The project's validation checks cover key production assumptions such as:

Six emotion classes
Model availability
Catalogue size
Six-way emotion features
Recommendation generation
SVM score normalization
Documented limitation status
⚠️ Important Limitation

The bundled music catalogue contains metadata, but it does not provide human-labelled song-emotion ground truth.

Therefore:

User text → emotion classification

is a supervised ML task with held-out evaluation.

Whereas:

Song metadata → emotion relevance

is a metadata/content-based recommendation signal.

So Moodify should not be interpreted as a scientifically validated system that objectively detects the emotional state of a song.

The 89.05% accuracy reported above refers specifically to user-text emotion classification.

🔬 Human Validation

The repository contains infrastructure for future human validation:

tools/annotation_app.py
tools/evaluate_human_validation.py
tools/create_annotation_queue.py

A 300-song annotation queue is included for:

English
Hindi
Punjabi

Future evaluation can include:

Accuracy
Balanced Accuracy
Macro-F1
Fleiss' Kappa
Precision@K
NDCG@K
Artist Diversity

Human song-emotion accuracy is not claimed until real annotations are collected.

🧩 Future Improvements
Human-validated song-emotion dataset
Lyrics-based emotion modelling
Audio-based emotion modelling
Multimodal metadata + lyrics + audio fusion
Stronger multilingual NLP
Transformer-based semantic retrieval
Personalized recommendation using user feedback
Learning-to-rank recommendation models
Recommendation diversity and novelty optimization
📌 Project Highlights
2,878
Songs

20,000
Emotion Text Examples

6
Emotion Classes

3
Languages

89.05%
Test Accuracy

84.38%
Macro-F1
👨‍💻 Author

Akshat Sajwan

B.Tech Computer Science student interested in:

Machine Learning · NLP · AI · Computer Vision · Python · Data Science

⭐ Project Summary

Moodify demonstrates a complete machine-learning application workflow:

Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
TF-IDF
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Emotion-Aware Ranking
 ↓
Recommendation
 ↓
Streamlit Deployment

The project focuses on building a practical recommendation system while keeping a clear distinction between validated model results and metadata-based recommendation assumptions.


This is the version I would use for the **fresh repository**. It is much more recruiter-friendly: the first screen immediately tells them **what Moodify is, what ML you used, the actual result, and how the recommendation works**, instead of making them scroll through a technical audit. The underlying project facts come from the project files you uploaded and the validated repository state. :contentReference[oaicite:0]{index=0}

After pasting it into GitHub's `README.md` editor, click **Preview** first. **Do not commit until you check the rendered page.**
