"""Offline integrity and recommendation smoke checks for Moodify."""
from pathlib import Path
import joblib
import pandas as pd
from src.emotion_data import EMOTIONS
from src.recommender import emotion_scores, recommend_by_emotion

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "models" / "moodify_model.joblib")
assert set(model.classes_) == set(EMOTIONS), model.classes_
cat = pd.read_csv(ROOT / "data" / "processed" / "labelled_catalogue.csv")
assert len(cat) == 2878
for e in EMOTIONS:
    assert f"emotion_{e}" in cat.columns
    assert f"emotion_prob_{e}" in cat.columns
    assert cat[f"emotion_{e}"].between(0, 1).all()
    assert cat[f"emotion_prob_{e}"].between(0, 1).all()
    assert len(recommend_by_emotion(cat, e, n=10)) == 10
    assert f"emotion_{e}" in cat.columns

# Prediction scores should sum to one and cover all six classes.
probs = emotion_scores(model, "I feel excited and happy and want to celebrate")
assert set(probs) == set(EMOTIONS)
assert abs(sum(probs.values()) - 1.0) < 1e-9
prob_cols = [f"emotion_prob_{e}" for e in EMOTIONS]
assert ((cat[prob_cols].sum(axis=1) - 1.0).abs() < 1e-5).all()
assert (ROOT / "reports" / "LIMITATION_STATUS.md").exists()
queue = pd.read_csv(ROOT / "reports" / "human_validation" / "annotation_queue_300.csv")
assert len(queue) == 300
assert queue["language"].value_counts().to_dict() == {"Hindi": 100, "Punjabi": 100, "English": 100}

print("MODEL_CLASSES_OK")
print("CATALOGUE_ROWS_OK", len(cat))
print("SIX_EMOTION_RECOMMENDATIONS_OK")
print("SVM_SCORE_NORMALIZATION_OK")
print("LIMITATION_STATUS_OK")
