# Moodify v4 — Limitation & Evidence Status

| Area | v4 status | What is actually fixed |
|---|---|---|
| Human ground-truth song emotions | **Workflow ready; data still required** | Local annotation app, instructions, 300-song balanced queue, majority vote + Fleiss' kappa evaluation. No labels are fabricated. |
| Lyrics | **Integration ready; source data still required** | Optional lyrics-derived six-way feature adapter. No copyrighted lyrics are bundled or reproduced. |
| Audio features | **Integration ready; source/model still required** | Optional audio-derived six-way feature adapter with strict ID joins. No audio features are invented. |
| Metadata-only inference | **Functionally upgradeable** | Recommender prefers a fused six-way profile when optional modalities exist; otherwise uses the verified metadata profile. |
| Human-proven recommendation quality | **Evaluation ready; annotations still required** | Precision@5/10, NDCG@5/10, artist diversity, majority agreement and Fleiss' kappa are computed on human-labelled songs only. |

## Important evidence boundary

The current bundled dataset does not contain human song-emotion labels, lyrics,
or audio features. Therefore this ZIP does **not** claim those limitations have
been magically solved by code. It supplies the complete integration and
validation machinery needed to add legitimate data without fabricating evidence.

For a portfolio/resume, describe the current catalogue emotion scores as
**metadata/content-based estimates** until real human labels and/or validated
multimodal models are supplied.


### v6 evidence gate
The recommender now applies a conservative 0.10 emotion-evidence floor before ranking. Rare language/emotion combinations may return fewer than the requested number of tracks rather than padding results with near-uniform metadata-only candidates. This improves precision at the cost of recall and does not create ground-truth song-emotion labels.
