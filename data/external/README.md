# Optional multimodal inputs

The bundled 2,878-song Moodify catalogue is metadata-only. This folder is an
explicit integration point for **legitimate, documented** lyrics-derived and
audio-derived features.

## Required schema

Each optional feature file must contain the exact Moodify catalogue `id` plus
six emotion probability columns:

- `lyrics_prob_sadness`, `lyrics_prob_joy`, `lyrics_prob_love`, `lyrics_prob_anger`, `lyrics_prob_fear`, `lyrics_prob_surprise`
- `audio_prob_sadness`, `audio_prob_joy`, `audio_prob_love`, `audio_prob_anger`, `audio_prob_fear`, `audio_prob_surprise`

Rows are joined **only by `id`**. Do not merge by title alone because remixes,
versions, duplicate titles, and artists can collide.

The probabilities must come from a documented model/source. Do not call them
human ground truth unless actual human annotations support that claim.

## Integration

```bash
python tools/prepare_multimodal_features.py \
  --lyrics data/external/lyrics_features.csv \
  --audio data/external/audio_features.csv
```

If only one modality is available, pass only that argument. Missing modalities
are not fabricated; available weights are renormalized. The recommender then
uses `fused_prob_<emotion>` when those columns exist.

## Human labels

Human ground truth belongs in `reports/human_validation/human_annotations.csv`
and is collected with `tools/annotation_app.py`. Three independent annotators
per song are recommended. The evaluation script refuses to fabricate metrics
when those annotations are absent.
