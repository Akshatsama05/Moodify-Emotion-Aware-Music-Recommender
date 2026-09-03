# Moodify Six-Emotion Model Report

## Production classes
The production classifier preserves the six original `dair-ai/emotion` classes: `sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`. No three-mood mapping is used in production.

## Held-out test evaluation
The final Linear SVM is fit on the 16,000 training rows plus 2,000 validation rows and evaluated once on the held-out 2,000-row test split.

- Accuracy: **0.8905**
- Balanced accuracy: **0.8526**
- Macro-F1: **0.8438**

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| sadness | 0.9351 | 0.9174 | 0.9262 | 581 |
| joy | 0.9294 | 0.9094 | 0.9193 | 695 |
| love | 0.7446 | 0.8616 | 0.7988 | 159 |
| anger | 0.8627 | 0.8909 | 0.8766 | 275 |
| fear | 0.8826 | 0.8393 | 0.8604 | 224 |
| surprise | 0.6667 | 0.6970 | 0.6815 | 66 |

These metrics measure **user-text emotion classification**, not objective song emotion recognition.

## Catalogue relevance correction
The supplied Spotify catalogue has no lyrics, audio features, or human emotion annotations. The corrected pipeline therefore does not pretend to create ground-truth song emotions. It computes six independent relevance scores from available song metadata using weighted lexical evidence and TF-IDF similarity to emotion prototypes. Low-confidence rows are marked `unclassified` rather than being forced into the first class.

This specifically removes the previous `0.05`-for-every-emotion plus `idxmax()` tie bias that caused un-evidenced tracks to become `sadness`.
