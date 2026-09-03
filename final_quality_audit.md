# Final Quality Audit — Corrected Build

## Result: PASS WITH DOCUMENTED LIMITATION

### Model
- Six original emotion classes preserved.
- Model artifact exposes all six classes.
- Held-out test accuracy: **0.8905**.
- Held-out balanced accuracy: **0.8526**.
- Held-out macro-F1: **0.8438**.

### Catalogue
- 2,878 tracks retained.
- English, Hindi and Punjabi catalogue coverage retained.
- Emotion scoring no longer uses a universal baseline followed by `idxmax()`.
- Lexical evidence is weighted and supplemented by TF-IDF similarity to six emotion prototypes.
- Artist identity is excluded from emotion scoring to reduce artist-specific bias.
- Low-confidence tracks are labelled `unclassified` rather than falsely assigned to an emotion.

### Recommendation path
`user text → six-class Linear SVM → selected emotion → metadata/content relevance → TF-IDF query similarity → popularity → ranked recommendations`

The Streamlit app does not hard-filter out low-confidence catalogue rows. This is deliberate: a missing emotion keyword should not make a track impossible to recommend. The selected emotion score remains the primary ranking signal.

### Limitation
Because the supplied catalogue contains metadata only, no audit can establish that a song *objectively* expresses an emotion. The system should be described as an **emotion-aware metadata/content recommender**, not a ground-truth music-emotion classifier.


## Remaining data limitation

The project cannot manufacture human song-emotion ground truth. To address this without a Google Form, the repository now includes an in-project annotation app, a reproducible 300-song balanced validation queue, and an evaluation script. These tools are ready, but human-validation metrics remain unavailable until real independent annotations are collected. The catalogue-side recommender no longer depends on the sparse `emotion_label`: it ranks using a complete six-way normalized emotion profile.


### v6 evidence gate
The recommender now applies a conservative 0.10 emotion-evidence floor before ranking. Rare language/emotion combinations may return fewer than the requested number of tracks rather than padding results with near-uniform metadata-only candidates. This improves precision at the cost of recall and does not create ground-truth song-emotion labels.
