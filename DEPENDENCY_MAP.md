# Moodify Six-Emotion Dependency Map

| Component | Depends on | Contract |
|---|---|---|
| `emotion_train.csv`, validation, test | original `dair-ai/emotion` splits | integer labels 0–5 mapped only to canonical six names |
| `src/emotion_data.py` | none | canonical `EMOTIONS` and label/name constants |
| `src/modeling.py` | scikit-learn | TF-IDF + balanced Linear SVM; final model classes are exactly the six emotions |
| `src/data.py` | raw Spotify metadata | normalized catalogue text and metadata fields |
| `src/labeling.py` | catalogue text + `EMOTIONS` | metadata/content-based relevance columns; explicitly not ground-truth song emotion |
| `src/recommender.py` | model output, catalogue relevance columns, TF-IDF index | `predict_emotion`, `emotion_scores`, and one shared `recommend_by_emotion` ranker |
| `data/processed/labelled_catalogue.csv` | `data/raw/spotify_metadata_catalogue.csv`, labeling module | 2,878 rows with `emotion_<class>` relevance signals and Spotify metadata |
| `models/moodify_model.joblib` | `src/modeling.py`, emotion splits | six-class production artifact consumed by `app.py` and Colab |
| `app.py` | all modules + artifact + processed catalogue | polished UI selection passes selected emotion directly to shared ranker; language/Spotify/deployment behavior retained |
| `Moodify_Master_Colab.ipynb` | all modules and project files | trains, evaluates, saves/verifies the same artifact, tests all six rankers, launches and health-checks app |

**Production path:** user text → six-class model → predicted emotion and scores → selected-emotion metadata/content relevance → TF-IDF similarity → popularity → ranked recommendations. No three-mood mapping is used.
