# Integration Verification — Corrected Six-Emotion Build

## Verified
- Production model exposes exactly six classes: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.
- Final test evaluation uses the held-out 2,000-row test split.
- Test accuracy: **0.8905**; balanced accuracy: **0.8526**; macro-F1: **0.8438**.
- Catalogue contains **2,878** tracks across English, Hindi and Punjabi metadata.
- Catalogue emotion scores are metadata/content relevance signals, not ground truth.
- Low-confidence catalogue rows are no longer forced into `sadness`.
- All six selected emotions can produce ranked recommendations from the complete eligible catalogue.
- Ranking remains emotion relevance + TF-IDF similarity + secondary popularity.
- Existing Streamlit UI and deployment flow are preserved.

## Catalogue label distribution after correction

| Label | Count |
|---|---:|
| sadness | 20 |
| joy | 41 |
| love | 167 |
| anger | 12 |
| fear | 9 |
| surprise | 6 |
| unclassified | 2,623 |

The `unclassified` bucket is intentional: metadata alone is insufficient evidence to assign a song an emotion. Recommendation ranking still considers every language-eligible track using the six emotion score columns, rather than discarding low-confidence tracks.


### v6 evidence gate
The recommender now applies a conservative 0.10 emotion-evidence floor before ranking. Rare language/emotion combinations may return fewer than the requested number of tracks rather than padding results with near-uniform metadata-only candidates. This improves precision at the cost of recall and does not create ground-truth song-emotion labels.
