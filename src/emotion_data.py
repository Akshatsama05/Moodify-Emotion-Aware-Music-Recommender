"""Canonical six-class emotion definitions for Moodify."""

EMOTION_NAMES = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
EMOTIONS = tuple(EMOTION_NAMES.values())
LABEL_TO_EMOTION = EMOTION_NAMES.copy()
EMOTION_TO_LABEL = {name: label for label, name in EMOTION_NAMES.items()}
