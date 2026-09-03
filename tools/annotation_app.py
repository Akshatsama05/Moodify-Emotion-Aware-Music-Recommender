"""Local Streamlit annotator for Moodify's 300-song validation queue.

Run from the project root:
    streamlit run tools/annotation_app.py
"""
from pathlib import Path
import html
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "reports" / "human_validation" / "annotation_queue_300.csv"
OUT = ROOT / "reports" / "human_validation" / "human_annotations.csv"
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise", "unclear"]

st.set_page_config(page_title="Moodify Human Validation", page_icon="🎧", layout="wide")
st.title("🎧 Moodify — Human Song-Emotion Validation")
st.caption("Independent validation of the catalogue emotion-ranking system. Do not use Moodify's predicted label when annotating.")

if not QUEUE.exists():
    st.error(f"Missing queue: {QUEUE}")
    st.stop()

queue = pd.read_csv(QUEUE)
if OUT.exists():
    annotations = pd.read_csv(OUT)
else:
    annotations = pd.DataFrame(columns=["annotator_id","song_id","emotion","confidence","intensity","notes"])

annotator = st.text_input("Annotator ID", value=st.session_state.get("annotator_id", ""), placeholder="A01")
if not annotator.strip():
    st.info("Enter an annotator ID to begin.")
    st.stop()
annotator = annotator.strip()
st.session_state["annotator_id"] = annotator

done = set(annotations.loc[annotations["annotator_id"].astype(str) == annotator, "song_id"].astype(str)) if not annotations.empty else set()
remaining = queue[~queue["song_id"].astype(str).isin(done)].copy()
st.progress((len(queue)-len(remaining))/max(len(queue),1), text=f"Completed: {len(queue)-len(remaining)} / {len(queue)}")
if remaining.empty:
    st.success("You have completed this validation queue.")
    st.stop()

row = remaining.iloc[0]
st.subheader(f"Song {row['queue_id']}")
st.write(f"**{row['song_name']}**")
st.write(f"Artist: {row['singer']}  ·  Album: {row['album']}  ·  Language: {row['language']}  ·  Genre: {row['genre']}")
if isinstance(row.get("spotify_url"), str) and row["spotify_url"].startswith("http"):
    st.link_button("Listen on Spotify ↗", row["spotify_url"])

emotion = st.radio("Dominant emotion", EMOTIONS, horizontal=True)
confidence = st.radio("Confidence", ["high", "medium", "low"], horizontal=True)
intensity = st.slider("Emotional intensity", 1, 5, 3)
notes = st.text_area("Optional note", placeholder="Why was this difficult or clear?")

if st.button("Save annotation", type="primary"):
    new = pd.DataFrame([{"annotator_id":annotator,"song_id":str(row["song_id"]),"emotion":emotion,"confidence":confidence,"intensity":intensity,"notes":notes.strip()}])
    annotations = pd.concat([annotations, new], ignore_index=True)
    annotations.to_csv(OUT, index=False)
    st.rerun()
