"""Reusable six-emotion model training and evaluation helpers."""
from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

def build_models() -> dict[str, Pipeline]:
    def vec(): return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)
    return {
        "Naive Bayes": Pipeline([("tfidf", vec()), ("model", MultinomialNB())]),
        "Logistic Regression": Pipeline([("tfidf", vec()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Linear SVM": Pipeline([("tfidf", vec()), ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))]),
    }

def build_final_model() -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True)), ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE))])

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    return train_test_split(df["text"], df["label"], test_size=test_size, random_state=RANDOM_STATE, stratify=df["label"])

def train_and_compare(df: pd.DataFrame):
    x_train, x_test, y_train, y_test = split_data(df)
    rows, fitted = [], {}
    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train); pred = pipeline.predict(x_test)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro")})
        fitted[name] = pipeline
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True), fitted, (x_test, y_test)

def evaluate_model(model, x_test, y_test) -> dict[str, Any]:
    pred = model.predict(x_test); labels = sorted(pd.unique(y_test))
    return {"predictions": pred, "accuracy": accuracy_score(y_test, pred), "macro_f1": f1_score(y_test, pred, average="macro"), "report": classification_report(y_test, pred, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_test, pred, labels=labels), "labels": labels}
