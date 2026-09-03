"""Validate and merge optional lyrics/audio emotion profiles into Moodify.

This script intentionally requires precomputed six-way probabilities. It does
not scrape lyrics, invent audio features, or infer human ground truth.
"""
from pathlib import Path
import argparse
import pandas as pd

from src.data import load_catalogue
from src.labeling import add_emotion_relevance
from src.multimodal import merge_optional_features, fuse_emotion_profiles
from src.emotion_data import EMOTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", default="data/raw/spotify_metadata_catalogue.csv")
    parser.add_argument("--lyrics", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", default="data/processed/labelled_catalogue_multimodal.csv")
    args = parser.parse_args()

    df = add_emotion_relevance(load_catalogue(args.catalogue))
    df = merge_optional_features(df, args.lyrics, args.audio)
    df = fuse_emotion_profiles(df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} songs to {out}")
    print("Six fused emotion profiles:", ", ".join(f"fused_prob_{e}" for e in EMOTIONS))


if __name__ == "__main__":
    main()
