"""Optional multimodal feature integration for Moodify.

Moodify's bundled catalogue is metadata-only. This module adds a strict,
opt-in path for externally supplied lyrics-derived and audio-derived emotion
scores. It never fabricates lyrics/audio data and never matches rows by title
alone. A song ID is required for safe joins.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .emotion_data import EMOTIONS


def _validate_feature_frame(frame: pd.DataFrame, name: str) -> None:
    if "id" not in frame.columns:
        raise ValueError(f"{name} must contain the exact catalogue 'id' column for safe joins.")
    if frame["id"].duplicated().any():
        dupes = frame.loc[frame["id"].duplicated(), "id"].astype(str).head(5).tolist()
        raise ValueError(f"{name} contains duplicate song ids: {dupes}")


def merge_optional_features(
    catalogue: pd.DataFrame,
    lyrics_path: str | Path | None = None,
    audio_path: str | Path | None = None,
) -> pd.DataFrame:
    """Left-join optional feature tables by catalogue song ``id`` only."""
    out = catalogue.copy()
    for path, name in ((lyrics_path, "lyrics features"), (audio_path, "audio features")):
        if not path:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        frame = pd.read_csv(path)
        _validate_feature_frame(frame, name)
        overlap = [c for c in frame.columns if c != "id" and c in out.columns]
        if overlap:
            frame = frame.rename(columns={c: f"{name.replace(' ', '_')}_{c}" for c in overlap})
        out = out.merge(frame, on="id", how="left", validate="one_to_one")
    return out


def _prob_columns(prefix: str) -> list[str]:
    return [f"{prefix}_prob_{emotion}" for emotion in EMOTIONS]


def fuse_emotion_profiles(
    df: pd.DataFrame,
    metadata_weight: float = 0.45,
    lyrics_weight: float = 0.30,
    audio_weight: float = 0.25,
) -> pd.DataFrame:
    """Fuse available six-way emotion profiles without inventing missing modalities.

    Expected optional columns are ``lyrics_prob_<emotion>`` and
    ``audio_prob_<emotion>``. Missing modalities are excluded and the remaining
    weights are renormalized per song. If no optional modality exists, the
    original metadata profile is retained.
    """
    out = df.copy()
    meta_cols = _prob_columns("emotion")
    if not all(c in out.columns for c in meta_cols):
        raise ValueError("Metadata emotion probability columns are missing.")

    modality_specs = [
        ("lyrics", lyrics_weight, _prob_columns("lyrics")),
        ("audio", audio_weight, _prob_columns("audio")),
    ]
    fused = np.zeros((len(out), len(EMOTIONS)), dtype=float)
    for i, emotion in enumerate(EMOTIONS):
        fused[:, i] = pd.to_numeric(out[f"emotion_prob_{emotion}"], errors="coerce").fillna(0).to_numpy()

    used_weight = np.full(len(out), metadata_weight, dtype=float)
    for _, weight, cols in modality_specs:
        if not all(c in out.columns for c in cols):
            continue
        values = out[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        available = np.isfinite(values).all(axis=1) & (values.sum(axis=1) > 0)
        if not available.any():
            continue
        values = np.nan_to_num(values, nan=0.0)
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
        fused[available] += weight * values[available]
        used_weight[available] += weight

    fused = fused / np.maximum(used_weight[:, None], 1e-12)
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-12)
    for i, emotion in enumerate(EMOTIONS):
        out[f"fused_prob_{emotion}"] = np.round(fused[:, i], 6)
    has_lyrics = all(c in out.columns for c in _prob_columns("lyrics"))
    has_audio = all(c in out.columns for c in _prob_columns("audio"))
    if has_lyrics or has_audio:
        availability = []
        for i in range(len(out)):
            parts = []
            if has_lyrics and pd.to_numeric(out.loc[out.index[i], _prob_columns("lyrics")], errors="coerce").notna().all():
                parts.append("lyrics")
            if has_audio and pd.to_numeric(out.loc[out.index[i], _prob_columns("audio")], errors="coerce").notna().all():
                parts.append("audio")
            availability.append("metadata + " + " + ".join(parts) if parts else "metadata-only fallback")
        out["emotion_profile_source"] = availability
    else:
        out["emotion_profile_source"] = "metadata-only fallback"
    return out
