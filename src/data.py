"""Data loading and cleaning utilities for the Moodify project."""
from __future__ import annotations

from pathlib import Path
import ast
import pandas as pd

REQUIRED_COLUMNS = [
    "id", "song_name", "singer", "album", "release_date", "cover_image",
    "spotify_url", "popularity", "genre", "language", "language_evidence", "source",
]


def load_catalogue(path: str | Path) -> pd.DataFrame:
    """Load the CSV and validate the columns required by the project."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return prepare_catalogue(df)


def _artist_text(value: object) -> str:
    """Turn the CSV's string representation of artist lists into readable text."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
    except (ValueError, SyntaxError):
        pass
    return text.strip("[]").replace("'", "")


def prepare_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text/date fields and add interpretable features used by Moodify."""
    out = df.copy()
    for col in ["song_name", "singer", "album", "genre", "language"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["artist_text"] = out["singer"].map(_artist_text)
    out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce")
    out["release_year"] = out["release_date"].dt.year.astype("Int64")
    out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").fillna(0)
    # Keep the original display/search text, but use a separate recommendation
    # text that excludes artist names. Artist names can otherwise create
    # accidental query similarity and leak identity into mood matching.
    out["text"] = (
        out["song_name"] + " " + out["album"] + " " + out["artist_text"] + " "
        + out["genre"] + " " + out["language"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    out["recommendation_text"] = (
        out["song_name"] + " " + out["album"] + " " + out["genre"] + " " + out["language"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return out.drop_duplicates(subset="id").reset_index(drop=True)


def catalogue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact missingness and cardinality audit for notebook use."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(dropna=False),
    }).sort_values(["missing", "unique"], ascending=[False, False])
