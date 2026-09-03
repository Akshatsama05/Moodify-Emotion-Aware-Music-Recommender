# Moodify — Six-Emotion Music Recommender

Moodify takes a short description of how a user feels, predicts one of six original `dair-ai/emotion` classes, and ranks songs from a curated Spotify metadata catalogue. The six production recommendation targets are **sadness, joy, love, anger, fear, and surprise**.

## Architecture

```text
User text → six-class Linear SVM → predicted emotion / decision scores
         → emotion-specific metadata/content relevance
         → TF-IDF content similarity → secondary popularity signal
         → ranked Spotify recommendations
```

The emotion classifier uses the preserved 16,000 / 2,000 / 2,000 train, validation, and test splits. It uses TF-IDF unigrams and bigrams (`min_df=2`, `max_features=12000`, `sublinear_tf=True`) and a balanced Linear SVM. The production artifact is `models/moodify_model.joblib` and must expose exactly the six emotion classes.

## Catalogue limitation

The catalogue contains metadata such as title, artist, album, genre, language, popularity, artwork, and Spotify URLs. Its supplied mood fields are empty. Accordingly, catalogue emotion is represented only as **metadata/content-based emotion relevance**, inferred by a transparent heuristic over available metadata. It is not ground-truth audio emotion, human annotation, or a claim about the emotional experience of a song. TF-IDF similarity to the user query and popularity are secondary ranking signals.

## Project structure and workflow

The polished Streamlit UI remains in `app.py`, with language controls, Spotify links, artwork, and the existing deployment flow preserved. Shared logic lives in `src/emotion_data.py`, `src/modeling.py`, `src/labeling.py`, `src/recommender.py`, and `src/data.py`. The matching end-to-end workflow is `Moodify_Master_Colab.ipynb`: upload/extract → dependencies → six-class training/evaluation → catalogue relevance preparation → shared recommendation ranking → artifact save/verify → Streamlit health check → temporary Cloudflare Quick Tunnel.

Run locally with `pip install -r requirements.txt` followed by `streamlit run app.py`.

## Corrected catalogue labeling

The original catalogue labeling implementation used the same baseline score for every emotion and then selected the first maximum, which could turn metadata with no emotion evidence into `sadness`. The corrected implementation removes that tie bias, strengthens multilingual lexical cues, uses comparable TF-IDF prototype similarity, and keeps low-confidence rows as `unclassified`. Artist names are excluded from mood/content scoring to reduce leakage. The Streamlit recommender ranks the full language-eligible catalogue so low-confidence metadata does not unnecessarily eliminate songs; shuffle samples only from the strongest candidates for the selected emotion.

## Limitations

The model labels user text using the six supervised emotion classes, but catalogue emotion relevance is metadata/content-based rather than ground truth. The catalogue lacks lyrics, audio features, and human song-emotion annotations in the supplied data, so relevance can miss context, sarcasm, multilingual nuance, and subjective interpretation. Catalogue emotion scores are therefore ranking signals, not validated song-emotion ground truth.

## Limitation status

See `reports/LIMITATION_STATUS.md` for an explicit distinction between limitations genuinely fixed in code and limitations that require new external data.

## Human validation (in-project, no Google Form)

The catalogue has no supplied human song-emotion labels, so Moodify does not claim psychological ground truth. The production catalogue now stores a complete six-way `emotion_prob_*` profile for every song, while `emotion_label` remains a conservative metadata-only label and may be `unclassified`.

A reproducible 300-song validation queue is included at `reports/human_validation/annotation_queue_300.csv` (100 English, 100 Hindi, 100 Punjabi). Use `tools/annotation_app.py` to annotate songs locally. Three independent annotators per song are recommended. Once annotations exist, `tools/evaluate_human_validation.py` computes catalogue emotion accuracy/balanced accuracy/macro-F1, confusion matrix, majority agreement, and recommendation Precision@K/NDCG@K plus artist diversity on the human-labelled subset. No human metrics are generated until actual annotations are supplied.

Run:

```bash
streamlit run tools/annotation_app.py
python tools/evaluate_human_validation.py
```

## Multimodal v4 integration

The project now has a safe multimodal integration layer in `src/multimodal.py`.
It can fuse six-way emotion profiles from metadata, lyrics-derived models and
audio-derived models when those inputs are legitimately available. The join
key is the catalogue `id`; title-only matching is deliberately rejected.

The bundled ZIP still contains **no fabricated lyrics, audio features, or human
labels**. Optional inputs belong under `data/external/`. Run
`tools/prepare_multimodal_features.py` after supplying documented feature
files. The app automatically prefers the resulting
`data/processed/labelled_catalogue_multimodal.csv` when present.

Human validation remains a real-data step: use `tools/annotation_app.py` and
then `tools/evaluate_human_validation.py`. The evaluator computes agreement,
classification metrics and recommendation Precision@K/NDCG/diversity only
from actual annotations; it never invents scores.

## Local Streamlit launch

For a normal local run, use `python run_local.py local` (or double-click `run_local.bat` on Windows). This opens Moodify at `http://127.0.0.1:8501` / `http://localhost:8501` and does not require Cloudflare, ngrok, or authentication. An optional ngrok mode is available with `python run_local.py ngrok` when `NGROK_AUTHTOKEN` is supplied. See `LOCAL_RUN_GUIDE.md`.


### v6 evidence gate
The recommender keeps a conservative 0.10 emotion-evidence floor as a first-tier preference. Strong-evidence tracks are ranked first; when fewer than 10 strong tracks are available, the highest-ranked remaining tracks are used only to complete the Top 10. This preserves recall without changing the 0.70 emotion / 0.20 similarity / 0.10 popularity ranking weights, and does not create ground-truth song-emotion labels.
