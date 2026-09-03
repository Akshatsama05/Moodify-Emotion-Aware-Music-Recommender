from pathlib import Path
import html
import random

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import load_catalogue
from src.emotion_data import EMOTIONS
from src.labeling import add_emotion_relevance
from src.recommender import predict_emotion, emotion_scores, recommend_by_emotion


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "moodify_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "labelled_catalogue_multimodal.csv"
BASE_DATA_PATH = ROOT / "data" / "processed" / "labelled_catalogue.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "spotify_metadata_catalogue.csv"

st.set_page_config(
    page_title="Moodify — Emotion-Aware Music",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --moodify-purple: #7c3aed; --moodify-blue: #2563eb; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
    .hero { padding: 2.2rem 2.4rem; border-radius: 28px; background: radial-gradient(circle at 85% 15%, rgba(96,165,250,.3), transparent 35%), linear-gradient(135deg, rgba(124,58,237,.20), rgba(37,99,235,.11)); border: 1px solid rgba(124,58,237,.20); margin-bottom: 1.5rem; box-shadow: 0 16px 45px rgba(76,29,149,.08); }
    .hero h1 { margin: .2rem 0 .35rem; font-size: clamp(2.25rem, 5vw, 4.2rem); letter-spacing: -.055em; }
    .hero p { max-width: 780px; margin: 0; font-size: 1.08rem; opacity: .78; line-height: 1.6; }
    .hero-kicker { font-size: .72rem; letter-spacing: .16em; font-weight: 800; opacity: .62; }
    .section-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; opacity: .58; margin: 1rem 0 .5rem; }
    .mood-panel { padding: 1rem 1.15rem; border-radius: 18px; background: linear-gradient(135deg, rgba(124,58,237,.12), rgba(59,130,246,.08)); border: 1px solid rgba(124,58,237,.16); }
    .song-card { height: 100%; padding: 0 0 1.1rem; border: 1px solid rgba(128,128,128,.18); border-radius: 20px; overflow: hidden; background: rgba(255,255,255,.035); box-shadow: 0 8px 24px rgba(15,23,42,.07); transition: transform .18s ease, box-shadow .18s ease; }
    .song-card:hover { transform: translateY(-3px); box-shadow: 0 14px 32px rgba(15,23,42,.13); }
    .song-card .cover { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; background: linear-gradient(135deg, #ddd6fe, #bfdbfe); }
    .song-card .song-body { padding: .95rem 1rem 0; }
    .song-rank { font-size: .72rem; font-weight: 800; color: var(--moodify-purple); letter-spacing: .08em; text-transform: uppercase; }
    .song-title { font-size: 1.05rem; font-weight: 800; line-height: 1.25; margin-top: .3rem; min-height: 2.65em; }
    .song-artist { opacity: .78; margin-top: .35rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .song-meta { font-size: .76rem; opacity: .62; margin-top: .7rem; min-height: 2.4em; }
    .empty-cover { width: 100%; aspect-ratio: 1 / 1; display:flex; align-items:center; justify-content:center; font-size: 3rem; background: linear-gradient(135deg, #ddd6fe, #bfdbfe); }
    div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.18); border-radius: 16px; padding: 12px; }
    @media (max-width: 700px) { .hero { padding: 1.5rem; } .song-title { font-size: .98rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

EMOTION_INFO = {"sadness": ("😢", "Sadness-relevant"), "joy": ("😊", "Joy-relevant"), "love": ("❤️", "Love-relevant"), "anger": ("😠", "Anger-relevant"), "fear": ("😨", "Fear-relevant"), "surprise": ("😲", "Surprise-relevant")}
EMOTION_OPTIONS = {"😢 Sadness": "sadness", "😊 Joy": "joy", "❤️ Love": "love", "😠 Anger": "anger", "😨 Fear": "fear", "😲 Surprise": "surprise"}
LANGUAGE_OPTIONS = {"All": None, "English": "English", "Hindi": "Hindi", "Punjabi": "Punjabi"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    # Prefer the multimodal catalogue when the optional integration pipeline
    # has produced it; otherwise use the verified metadata-only catalogue.
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if BASE_DATA_PATH.exists():
        return pd.read_csv(BASE_DATA_PATH)
    return add_emotion_relevance(load_catalogue(RAW_DATA_PATH))


@st.cache_resource
def build_song_index(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df["recommendation_text"].fillna(df["text"]).fillna(""))
    return vectorizer, matrix


def safe_text(value, fallback=""):
    text = "" if pd.isna(value) else str(value).strip()
    return text or fallback


def song_card(row, rank: int):
    cover = safe_text(row.get("cover_image"))
    title = html.escape(safe_text(row.get("song_name"), "Unknown song"))
    artist = html.escape(safe_text(row.get("artist_text"), "Unknown artist"))
    language = html.escape(safe_text(row.get("language")))
    genre = html.escape(safe_text(row.get("genre")))
    album = html.escape(safe_text(row.get("album")))
    popularity = int(float(row.get("popularity", 0) or 0))
    spotify = safe_text(row.get("spotify_url"))
    meta = " · ".join(part for part in [language, genre] if part) or "Mood-matched track"
    if album:
        meta = f"{meta}<br><span title=\"{album}\">{album}</span>"

    st.markdown(
        f"""
        <div class="song-card">
          {f'<img class="cover" src="{html.escape(cover, quote=True)}" alt="Album artwork for {title}">' if cover.startswith('http') else '<div class="empty-cover">🎵</div>'}
          <div class="song-body">
            <div class="song-rank">#{rank} · Popularity {popularity}/100</div>
            <div class="song-title">{title}</div>
            <div class="song-artist">{artist}</div>
            <div class="song-meta">{meta}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if spotify.startswith("http"):
        st.link_button("Open on Spotify ↗", spotify, use_container_width=True)
    else:
        st.caption("Spotify link unavailable")


def language_pool(df: pd.DataFrame, mood: str, language: str | None) -> pd.DataFrame:
    pool = df[df["mood"].fillna("").str.lower() == str(mood).lower()].copy()
    if language:
        pool = pool[pool["language"].fillna("").str.casefold() == language.casefold()]
    return pool


def render_recommendations(recs: pd.DataFrame, shuffled: bool):
    if recs.empty:
        st.info("No songs are available for this mood and language combination yet. Try **All** or another language.")
        return
    actual = len(recs)
    mode = "Shuffled picks" if shuffled else "Top 10 matches"
    st.markdown(f"### {mode}")
    st.caption(f"Showing {actual} Top 10 match{'es' if actual != 1 else ''}. Strong emotion-evidence tracks are ranked first; lower-evidence tracks are used only when needed to complete the list.")
    for start in range(0, actual, 4):
        cols = st.columns(4, gap="medium")
        for col, (rank, (_, row)) in zip(cols, enumerate(recs.iloc[start:start + 4].iterrows(), start=start + 1)):
            with col:
                song_card(row, rank)


st.markdown(
    "<div class='hero'><div class='hero-kicker'>NLP • MACHINE LEARNING • MUSIC</div><h1>🎧 Moodify</h1><p>Describe the feeling you want to soundtrack. Moodify predicts one of six emotions and turns it into a polished, language-aware Spotify playlist direction.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Playlist controls")
    language_label = st.selectbox("Language", list(LANGUAGE_OPTIONS), index=0)
    st.divider()
    st.header("About Moodify")
    st.write(
        "**NLP:** TF-IDF text representation  \n"
        "**ML:** Linear SVM six-emotion classifier  \n"
        "**Recommendation:** emotion relevance + TF-IDF similarity + popularity ranking"
    )
    st.caption("The controls change only the recommendation presentation and selection. The trained model remains unchanged.")

model = load_model()
catalogue = load_data()
vectorizer, song_matrix = build_song_index(catalogue)

st.markdown("<div class='section-label'>Step 1 · Describe your feeling</div>", unsafe_allow_html=True)
examples = [
    "I feel peaceful after a long day and want something soft",
    "I am excited and want an energetic song for a party",
    "I feel lonely and nostalgic and want something emotional",
]
example = st.selectbox("Try an example", ["Write my own"] + examples, label_visibility="collapsed")
default_text = "" if example == "Write my own" else example
user_text = st.text_area(
    "Describe your mood in a sentence",
    value=default_text,
    height=110,
    placeholder="e.g., I had a rough day and want a quiet, emotional song",
)
selected_emotion_display = st.selectbox(
    "Choose a Moodify emotion",
    list(EMOTION_OPTIONS),
    help="These are the six original dair-ai/emotion labels. Your choice is used directly to rank emotion-relevant songs.",
)
selected_emotion = EMOTION_OPTIONS[selected_emotion_display]

analyze = st.button("Analyze mood", type="primary", use_container_width=True)
if analyze:
    if len(user_text.strip()) < 3:
        st.warning("Please enter a little more about how you're feeling.")
        st.stop()
    st.session_state["moodify_text"] = user_text.strip()
    st.session_state["moodify_emotion"] = predict_emotion(model, user_text.strip())
    st.session_state["moodify_scores"] = emotion_scores(model, user_text.strip())
    st.session_state["moodify_shuffle"] = False
    st.session_state["moodify_shuffle_nonce"] = 0

if "moodify_emotion" in st.session_state:
    predicted = st.session_state["moodify_emotion"]
    query = st.session_state["moodify_text"]
    selected_score = st.session_state.get("moodify_scores", {}).get(selected_emotion, 0.0)
    icon, label = EMOTION_INFO[selected_emotion]
    st.markdown("<div class='section-label'>Step 2 · Your mood direction</div>", unsafe_allow_html=True)
    metric_col, info_col = st.columns([1, 3])
    with metric_col:
        st.metric("Selected emotion", selected_emotion_display)
    with info_col:
        st.markdown(f"<div class='mood-panel'><strong>{selected_emotion_display}.</strong><br>Recommendations use {label.lower()} metadata/content relevance, TF-IDF similarity, and popularity.</div>", unsafe_allow_html=True)
    st.caption(f"Model prediction: {predicted.title()} (selected emotion score: {selected_score:.3f}). The selected emotion directly controls ranking.")

    # Do not pre-filter by a hard emotion-label threshold. Low-confidence catalogue
    # rows are still eligible and are ranked by the selected emotion score,
    # query similarity, and popularity. This avoids throwing away most of the
    # catalogue simply because metadata lacks explicit emotion words.
    pool = catalogue.copy()
    if LANGUAGE_OPTIONS[language_label]: pool = pool[pool["language"].fillna("").str.casefold() == LANGUAGE_OPTIONS[language_label].casefold()]
    shuffled = bool(st.session_state.get("moodify_shuffle", False))
    shuffle_clicked = st.button("🔀 Shuffle Music", use_container_width=True)
    if shuffle_clicked:
        st.session_state["moodify_shuffle"] = True
        st.session_state["moodify_shuffle_nonce"] = st.session_state.get("moodify_shuffle_nonce", 0) + 1
        shuffled = True

    if shuffled:
        # Shuffle only among strong candidates for the selected emotion.
        # The old implementation sampled the entire catalogue, which could
        # return songs unrelated to the selected mood.
        candidate_count = 10
        candidate_pool = recommend_by_emotion(
            pool, selected_emotion, n=candidate_count, query=query,
            vectorizer=vectorizer, song_matrix=song_matrix
        )
        recs = candidate_pool.sample(
            frac=1, random_state=random.SystemRandom().randint(0, 2**32 - 1)
        )
    else:
        # Keep the original normal recommendation ranking untouched.
        recs = recommend_by_emotion(
            catalogue if LANGUAGE_OPTIONS[language_label] is None else pool,
            selected_emotion,
            n=10,
            query=query,
            vectorizer=vectorizer,
            song_matrix=song_matrix,
        )
    render_recommendations(recs, shuffled)

    with st.expander("How did the NLP model make this prediction?"):
        st.write(
            "Moodify converts the input text into TF-IDF features and sends those features through the unchanged trained Linear SVM classifier. The selected emotion directly controls metadata/content relevance scoring; TF-IDF similarity and popularity provide secondary ranking signals."
        )
else:
    st.info("Analyze your mood to unlock a personalized playlist direction.")

with st.expander("Project notes & limitations"):
    st.markdown(
        """
        - The catalogue contains **2,878 songs** across English, Hindi/Bollywood and Punjabi music.
        - The emotion model is trained **only on `dair-ai/emotion`** (16,000 train / 2,000 validation / 2,000 test) and preserves all six original emotion classes.
        - The Spotify catalogue has transparent **metadata/content-based emotion relevance scores** because its supplied mood fields are empty. These are ranking signals, not ground-truth audio or human song-emotion labels.
        - Warriner et al. (2013) VAD norms are an external mapping-validation reference, not a supervised training dataset.
        - The emotion-model metrics measure performance on the six original `dair-ai/emotion` labels, not human-validated psychological song-emotion recognition.
        """
    )
