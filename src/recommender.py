"""Shared six-emotion prediction and metadata/content relevance ranking."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .emotion_data import EMOTIONS

def predict_emotion(model, text: str) -> str:
    return str(model.predict([text])[0])

def emotion_scores(model, text: str) -> dict[str, float]:
    """Return softmax-normalized class scores from the Linear SVM decision values.

    These are probability-like ranking scores, not calibrated probabilities.
    """
    scores = {e: 0.0 for e in EMOTIONS}
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function([text]))[0]
        classes = list(model.classes_)
        raw = raw - np.max(raw)
        probs = np.exp(raw)
        probs = probs / probs.sum()
        for e, value in zip(classes, probs):
            scores[str(e)] = float(value)
    else:
        scores[predict_emotion(model, text)] = 1.0
    return scores

MIN_EMOTION_EVIDENCE = 0.10

def recommend_by_emotion(df: pd.DataFrame, emotion: str, n: int = 10, query: str | None = None, vectorizer: TfidfVectorizer | None = None, song_matrix=None) -> pd.DataFrame:
    if emotion not in EMOTIONS: raise ValueError(f"Unknown emotion: {emotion}")
    score_col = (f"fused_prob_{emotion}" if f"fused_prob_{emotion}" in df.columns else (f"emotion_prob_{emotion}" if f"emotion_prob_{emotion}" in df.columns else f"emotion_{emotion}"))
    evidence_col = f"emotion_{emotion}" if f"emotion_{emotion}" in df.columns else score_col
    required = {score_col, evidence_col, "popularity", "song_name", "artist_text", "genre", "language", "spotify_url"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Dataframe is missing recommendation columns: {sorted(missing)}")
    result = df.copy()
    if query and vectorizer is not None and song_matrix is not None:
        q = vectorizer.transform([query])
        # DataFrame filtering preserves the original row index, so this remains
        # aligned with the precomputed sparse matrix.
        sims = cosine_similarity(q, song_matrix[result.index]).ravel()
        result["similarity"] = sims
    else:
        result["similarity"] = 0.0
    result["emotion_score"] = pd.to_numeric(result[score_col], errors="coerce").fillna(0.0)
    result["emotion_evidence"] = pd.to_numeric(result[evidence_col], errors="coerce").fillna(0.0)
    result["popularity_score"] = pd.to_numeric(result["popularity"], errors="coerce").fillna(0.0).clip(0, 100) / 100.0
    # Emotion is intentionally dominant. Similarity personalizes within the
    # emotion, while popularity is only a small tie-breaker.
    #
    # Keep the conservative evidence gate as the first-tier preference, but do
    # not let it truncate the requested Top-N list. If fewer than n tracks have
    # strong evidence, the highest-ranked remaining tracks are used only to
    # complete the list. This preserves the same scoring/ranking system while
    # avoiding a 5-song result when ten catalogue tracks are available.
    result["rank_score"] = (
        0.70 * result["emotion_score"]
        + 0.20 * result["similarity"]
        + 0.10 * result["popularity_score"]
    )
    result = result.sort_values(["rank_score", "emotion_score", "similarity", "popularity"], ascending=False)
    # Remove only true metadata duplicates (same title + artist), not legitimate
    # different songs that happen to share a title.
    result = result.drop_duplicates(subset=["song_name", "artist_text"], keep="first")

    strong = result[result["emotion_evidence"] >= MIN_EMOTION_EVIDENCE].copy()
    if len(strong) >= n:
        return strong.head(n)

    # Strong-evidence tracks always come first. Only when there are not enough
    # of them do we use the next-best scored tracks to complete Top-N.
    weak = result[result["emotion_evidence"] < MIN_EMOTION_EVIDENCE].copy()
    return pd.concat([strong, weak], ignore_index=False).head(n)

def explain_prediction(model, song_text: str, top_n: int = 8) -> pd.DataFrame:
    estimator = model.named_steps["model"]; vectorizer = model.named_steps["tfidf"]
    if not hasattr(estimator, "coef_"): return pd.DataFrame(columns=["term", "importance"])
    row = vectorizer.transform([song_text]); names = vectorizer.get_feature_names_out(); predicted = model.predict([song_text])[0]
    index = list(estimator.classes_).index(predicted); scores = row.toarray()[0] * estimator.coef_[index]; top = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": names[top], "importance": scores[top]})
