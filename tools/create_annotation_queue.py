"""Create a reproducible, language-balanced human-validation queue."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "data" / "processed" / "labelled_catalogue.csv"
OUT = ROOT / "reports" / "human_validation" / "annotation_queue_300.csv"

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
df = pd.read_csv(CAT)
# 100 per catalogue language, distributed across popularity quartiles.
parts = []
for language, group in df.groupby("language", sort=True):
    group = group.copy()
    group["pop_bin"] = pd.qcut(group["popularity"].rank(method="first"), 4, labels=False)
    chunks = []
    for _, q in group.groupby("pop_bin", sort=True):
        chunks.append(q.sample(n=min(25, len(q)), random_state=20260902))
    part = pd.concat(chunks).sample(n=min(100, len(group)), random_state=20260902)
    parts.append(part)
queue = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=20260902).reset_index(drop=True)
queue["queue_id"] = [f"HV{i:04d}" for i in range(1, len(queue)+1)]
cols = ["queue_id", "id", "song_name", "singer", "album", "genre", "language", "popularity", "spotify_url"]
queue[cols].rename(columns={"id":"song_id"}).to_csv(OUT, index=False)
print(f"Wrote {len(queue)} songs to {OUT}")
