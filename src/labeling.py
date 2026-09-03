"""Metadata/content-based emotion relevance for catalogue ranking.

The catalogue has no supplied lyrics, audio features, or human song-emotion
annotations.  Moodify therefore computes transparent *relevance scores* from
available metadata.  Exact lexical evidence is combined with TF-IDF similarity
to short emotion prototype descriptions.  Songs without enough evidence are
marked ``unclassified`` instead of being forced into the first emotion.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .emotion_data import EMOTIONS

# Romanised Hindi/Punjabi terms are included because the catalogue contains
# Hindi/Bollywood and Punjabi metadata.  These are ranking cues, not labels.
EMOTION_KEYWORDS = {
    "sadness": "sad alone lonely tears goodbye broken pain loss missing sorrow cry heartache hurt grief regret melancholy udaas udasi dukh dukhi tanha tanhai judai judaai aansu rona rooh bewafa bewafai dard akela akeli akelapan bichhadna bichhda vichhora vichhoda hanju hanjoo rondi rovna kalla kalli kalliyan tuttia tuteya dil yaad yaadan gam gham udaasi",
    "joy": "happy happiness fun party dance celebration celebrate smile good life sunshine joy joyful excited energy energetic laugh laughter victory win festive festival khushi khush nach nacho jashn masti balle balle bhangra utsav jeet khushiyan hass hassna hasna nachdi nachde rangila rangili mauj maujan chardi kala",
    "love": "love lover loved loving romantic romance kiss heart valentine darling baby together forever affection adore ishq pyaar pyar mohabbat prem saath saathiya sanam jaan mahi sohniya heer ranjha tera meri mera hum tum ishqiya milna milan yaara yaari",
    "anger": "anger angry rage fight revenge hate frustrated rebel riot fury mad furious hostility conflict resentment gussa gusse ghussa kroodh nafrat badla jang dushmani vair vairi ranjish rosh krodh inteqam intikaam qatal maar tod phaad ladayi ladaai jhagra jhagda zulm baghawat bagawat",
    "fear": "fear afraid scared danger nightmare ghost panic anxious warning terror horror haunted khauf darr bhay bhoot andhera saaya saya scary maut marna khatra khatarnak pret chudail aatma aatank ghabrahat wehshat",
    "surprise": "surprise surprised wow unexpected shock wonder magic sudden twist unbelievable amazed astonish astonishment hairan achanak ajab adbhut wah ohho omg kamaal kamal hairani heran chamatkar jadoo jaadu anokha anokhi shocking unexpected ulta",
}

# Language-aware prototypes keep Romanized Hindi/Punjabi metadata from being
# compared only against English descriptions. They are semantic cues, not labels.
EMOTION_PROTOTYPES = {
    "sadness": "sadness melancholy grief heartbreak sorrow loneliness loss regret tears goodbye emotional pain missing someone udaas udasi dukh dukhi tanhai judai judaai aansu dard bewafa bewafai yaad gham gam hanju vichhora kalla",
    "joy": "joy happiness celebration excitement cheerful fun party dancing smiling laughter energy victory festive upbeat khushi khushiyan masti jashn nach bhangra hassna chardi kala",
    "love": "love romance affection relationship devotion beloved heart kiss together forever passion ishq pyaar pyar mohabbat prem saathiya sanam jaan mahi heer ranjha yaara",
    "anger": "anger rage hostility conflict frustration hatred revenge fighting rebellion fury aggressive resentment gussa ghussa nafrat badla dushmani vair krodh kroodh qatal ladayi jhagra zulm baghawat",
    "fear": "fear anxiety terror horror danger nightmare darkness ghost panic warning threat scared uncertainty dread darr khauf bhay bhoot saaya andhera maut khatra ghabrahat",
    "surprise": "surprise astonishment amazement unexpected shock wonder sudden twist magic unbelievable wow mysterious hairan achanak ajab adbhut kamaal kamal jadoo anokha chamatkar",
}

# Strong multi-word title phrases. Phrase hits outrank weak single-word cues.
PHRASE_CUES = {
    "sadness": [
        "broken heart", "broken hearts", "never be lonely", "would you rather be lonely",
        "one more day won", "flow my tears", "the way we say goodbye",
        "bewafa tera", "teri hogaiyaan", "teri hogayi", "tum bin", "bin tere",
        "judai", "judaai", "so dukh", "dukh kaisa", "bewafai kar gaya",
        "broken but beautiful", "aabaad barbaad", "agar tum saath ho",
        "channa mereya", "hamari adhuri kahani", "phir bhi tumko chahunga",
        "tujhe kitna chahne", "tadap tadap", "jiya dhadak dhadak",
        "kya mujhe pyaar hai", "main dhoondne ko", "alvida", "yaad aa rahi",
        "hanju", "vichhora", "vichhoda", "kalli", "kalliyan", "kalla",
    ],
    "joy": [
        "party on my mind", "party all night", "one two three four", "dance basanti",
        "lungi dance", "happy family", "too much fun", "are we having any fun",
        "sure feels good", "sauda khara khara", "daaru party", "this party",
        "dance like", "good luck", "khush reha kar", "enna khush rakhuga",
        "balle balle", "nach meri rani", "gallaan goodiyaan", "aaj ki raat",
    ],
    "love": [
        "ishq wala love", "phir mohabbat", "pehla pyaar", "tere pyaar mein",
        "ishq bulaava", "ye ishq hai", "ishq tera", "filhaal2 mohabbat",
        "one love", "nira ishq", "saada pyaar", "pyaar hoya", "dil tu jaan tu",
        "at most a kiss", "are you in love", "heart on my sleeve", "wild heart",
        "let's leave together", "so long forever",
    ],
    "anger": [
        "fight on", "hate the player", "mad mind", "revenge", "red flags",
        "i hate luv storys", "hai dil ye mera", "qatal", "king shit", "winning speech",
        "case", "mvp", "no love", "dushmani", "badla", "inteqaam", "ladayi", "jhagra",
    ],
    "fear": [
        "your ghost", "i saw a ghost", "red flags & warning signs", "panic like",
        "channa ve", "bhoot", "nightmare", "haunted", "khauf", "darr", "bhoot bangla",
        "saaya", "andhera", "khatra", "aatank",
    ],
    "surprise": [
        "black magic", "wow", "ajab si", "twist", "unexpected", "sudden",
        "what a", "oh my", "omg", "kamaal", "ajab", "achanak", "magic",
    ],
}


# A small set of genre cues is useful only as a weak prior.  The catalogue's
# genre field has three broad buckets, so it is deliberately low-weight.
GENRE_CUES = {
    "joy": {"party", "dance", "disco", "edm", "celebration"},
    "anger": {"metal", "punk", "hardcore", "rock"},
}


def _tokens(value) -> set[str]:
    return set(re.findall(r"[a-z]+", str(value).lower()))


def _metadata_text(df: pd.DataFrame) -> pd.Series:
    """Build emotion-scoring text without artist names to reduce artist bias."""
    cols = [c for c in ("song_name", "album", "genre", "language") if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    # Title is the strongest available signal; album/genre/language provide
    # weaker context without using artist identity as an emotion cue.
    title = df["song_name"].fillna("").astype(str) if "song_name" in df.columns else pd.Series([""] * len(df), index=df.index)
    album = df["album"].fillna("").astype(str) if "album" in df.columns else pd.Series([""] * len(df), index=df.index)
    context = df[[c for c in ("genre", "language") if c in df.columns]].fillna("").astype(str).agg(" ".join, axis=1) if any(c in df.columns for c in ("genre", "language")) else pd.Series([""] * len(df), index=df.index)
    text = title + " " + title + " " + title + " " + title + " " + album + " " + context
    return text.str.lower()


def _lexical_scores(texts: pd.Series) -> np.ndarray:
    """Score explicit title/metadata evidence with phrase-aware language cues."""
    out = np.zeros((len(texts), len(EMOTIONS)), dtype=float)
    for i, raw in enumerate(texts.astype(str)):
        text = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        tokens = set(text.split())
        for j, emotion in enumerate(EMOTIONS):
            words = _tokens(EMOTION_KEYWORDS[emotion])
            hits = len(tokens & words)
            phrase_hits = sum(1 for p in PHRASE_CUES.get(emotion, []) if p in text)
            # Phrase evidence is stronger, while multiple individual cues add
            # support without letting long metadata fields dominate.
            score = min(0.72, 0.14 * hits) + min(0.55, 0.38 * phrase_hits)
            out[i, j] = min(1.0, score)
    return out


def _semantic_scores(texts: pd.Series) -> np.ndarray:
    """Compute consistent prototype similarity using one fixed vocabulary.

    The previous implementation refit TF-IDF for every input row, which made
    IDF weights depend on the individual song. Fitting once on the complete
    catalogue text plus prototypes makes scores comparable across songs.
    """
    prototypes = [EMOTION_PROTOTYPES[e] for e in EMOTIONS]
    corpus = list(texts.astype(str)) + prototypes
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[: len(texts)], matrix[len(texts) :])


def emotion_relevance(text: str, emotion: str) -> float:
    """Return a deterministic 0..1 relevance score for one text/emotion pair."""
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion: {emotion}")
    single = pd.Series([str(text)])
    lexical = _lexical_scores(single)[0, list(EMOTIONS).index(emotion)]
    semantic = _semantic_scores(single)[0, list(EMOTIONS).index(emotion)]
    return float(np.clip(0.70 * lexical + 0.30 * semantic, 0.0, 1.0))


def add_emotion_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Add six-way emotion relevance and normalized emotion profiles.

    The catalogue has no human song-emotion labels, so these are explicitly
    content/metadata estimates.  Every song receives a six-emotion profile
    (``emotion_prob_*``) so unclassified rows are still rankable;
    ``emotion_label`` is only a conservative display label.
    """
    out = df.copy()
    texts = _metadata_text(out)
    lexical = _lexical_scores(out["song_name"].fillna("") if "song_name" in out.columns else texts)
    semantic = _semantic_scores(texts)
    scores = 0.70 * lexical + 0.30 * semantic

    if "genre" in out.columns:
        genres = out["genre"].fillna("").astype(str).str.lower()
        for j, emotion in enumerate(EMOTIONS):
            cues = GENRE_CUES.get(emotion, set())
            if cues:
                scores[:, j] += genres.apply(
                    lambda g: 0.03 if any(c in g for c in cues) else 0.0
                ).to_numpy()

    scores = np.clip(scores, 0.0, 1.0)
    for j, emotion in enumerate(EMOTIONS):
        out[f"emotion_{emotion}"] = scores[:, j]

    # Convert the six raw relevance signals into a comparable six-way profile.
    # A temperature keeps weak metadata from becoming overconfident while
    # still giving songs with evidence a useful ranking signal.
    temperature = 0.08
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted / temperature)
    probs = exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)
    for j, emotion in enumerate(EMOTIONS):
        out[f"emotion_prob_{emotion}"] = probs[:, j].round(6)

    score_frame = pd.DataFrame(scores, columns=EMOTIONS, index=out.index)
    best = score_frame.idxmax(axis=1)
    confidence = score_frame.max(axis=1)
    # Conservative label for display/reporting only. The recommender uses the
    # complete six-way profile instead of filtering on this label.
    out["emotion_label"] = np.where(confidence >= 0.10, best, "unclassified")
    out["emotion_confidence"] = confidence.round(6)
    out["emotion_label_source"] = (
        "metadata/content relevance: weighted lexical evidence + TF-IDF emotion prototype similarity; "
        "weak genre cue; not ground-truth song emotion"
    )
    return out


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    labels = list(EMOTIONS) + (["unclassified"] if "unclassified" in set(df["emotion_label"]) else [])
    return df["emotion_label"].value_counts().reindex(labels, fill_value=0).rename_axis("emotion").reset_index(name="count")
