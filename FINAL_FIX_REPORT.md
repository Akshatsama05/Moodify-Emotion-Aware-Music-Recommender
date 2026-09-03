# Moodify — Final Engineering Fix Report

## Fixed

1. **Catalogue sadness tie-bias**
   - Removed the universal baseline + first-maximum behavior that assigned unsupported songs to `sadness`.
   - Low-evidence tracks remain `unclassified`.

2. **Comparable emotion relevance scoring**
   - Catalogue emotion relevance combines weighted lexical evidence with TF-IDF prototype similarity.
   - Prototype TF-IDF is fit consistently over the catalogue and emotion prototypes rather than refit per song.

3. **Artist leakage in mood scoring**
   - Artist names are excluded from catalogue emotion scoring.
   - Recommendation similarity uses `recommendation_text` without artist identity.

4. **Weak/ambiguous anger cues**
   - Removed especially ambiguous lexical cues such as `war` and `violence` from the anger keyword set.
   - Metadata-only scoring remains explicitly heuristic.

5. **SVM decision-score handling**
   - The app now softmax-normalizes Linear SVM decision scores into probability-like class weights that sum to 1.
   - These are not claimed to be calibrated probabilities.

6. **Shuffle bug**
   - Previously, Shuffle sampled the entire catalogue and could return tracks unrelated to the selected emotion.
   - Shuffle now samples only from the strongest candidates for the selected emotion and language.

7. **Duplicate recommendation titles**
   - Recommendation ranking removes duplicate song titles so repeated catalogue entries do not consume the requested recommendation slots.

8. **Regression checks**
   - Added `validate_project.py` to verify model classes, catalogue size, six emotion score columns, six-emotion recommendation smoke tests, and normalized model scores.
   - Python compilation checks pass.

## Current honest limitation

The supplied 2,878-track catalogue contains metadata but no lyrics, audio features, or human song-emotion annotations. Therefore catalogue emotion values are **metadata/content relevance signals**, not validated ground-truth song-emotion labels. The six-class supervised metrics apply to the `dair-ai/emotion` user-text classification task, not to objective recognition of song emotion.


### v6 evidence gate
The recommender now applies a conservative 0.10 emotion-evidence floor before ranking. Rare language/emotion combinations may return fewer than the requested number of tracks rather than padding results with near-uniform metadata-only candidates. This improves precision at the cost of recall and does not create ground-truth song-emotion labels.

## v6 emotion-quality upgrade

v6 strengthens the metadata-only catalogue layer without fabricating song-emotion labels:
- expanded Hindi/Romanized-Hindi and Punjabi/Romanized-Punjabi emotion vocabulary;
- added phrase-level cues for all six emotions;
- added language-aware emotion prototype text for semantic matching;
- title is weighted more heavily than album/genre/language metadata;
- recommender applies a 0.10 raw emotion-evidence gate so low-evidence songs are not padded into rare emotion playlists;
- recommendation ranking remains emotion-dominant, with TF-IDF similarity and popularity as secondary signals;
- the audit now covers all 18 language × emotion combinations.

The evidence gate intentionally allows fewer than 10 recommendations for sparse combinations. This is a quality/precision safeguard, not a claim of human-validated song emotion.
