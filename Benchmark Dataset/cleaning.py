# cleaning.py

import sys
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st




if len(sys.argv) > 1:
    data_path = Path(sys.argv[1]).expanduser().resolve()
else:
    st.error("Usage: python cleaning.py data.tsv")
    st.stop()
img_root = data_path.parent


def parse_list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)

def parse_json(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x.replace('""', '"'))
    except json.JSONDecodeError:
        return ast.literal_eval(x)

def distil_centres(traj):
    pts = []
    for actor in traj:
        for bb in actor.get("bbox_per_frame", []):
            if "frame" in bb and "mid_x" in bb and "mid_y" in bb:
                pts.append((bb["frame"], bb["mid_x"], bb["mid_y"]))
    return sorted(pts, key=lambda t: t[0])

def draw_trajectory(img, centres):
    if len(centres) < 2:
        return
    points = [(int(x), int(y)) for _, x, y in centres]
    for p, q in zip(points, points[1:]):
        cv2.line(img, p, q, (0, 0, 255), 3)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    for col in ["frame_numbers", "object_bbox", "trajectory_bbox"]:
        df[col] = df[col].apply(parse_list)
    df["trajectory_raw"] = df["trajectory"].apply(parse_json)
    df["trajectory"] = df["trajectory_raw"].apply(distil_centres)
    return df

df = load_data(data_path)
total = len(df)

# ────────────────────────────────
# Session state for navigation
# ────────────────────────────────
state = st.session_state
if "idx" not in state:
    state.idx = 1420
if "exclude" not in state:
    state.exclude = []


st.title("VIRAT Vehicle Event Filter")

names = df["img_best"] \
    .astype(str) \
    .apply(Path) \
    .apply(lambda p: p.name) \
    .tolist()

with st.sidebar:
    st.markdown("### Jump to Image")
    sel = st.selectbox("Image:", options=range(total), format_func=lambda i: names[i], index=state.idx)
    if sel != state.idx:
        state.idx = sel

i = state.idx
st.caption(f"Item {i+1}/{total}")

if i >= total:
    st.success("Done!")
    if st.button("Export excluded"):
        if state.exclude:
            out = df.iloc[state.exclude]
            out.to_csv("excluded_cases.tsv", sep="\t", index=False)
            st.info("Wrote excluded_cases.tsv")
        else:
            st.info("No exclusions.")
    st.stop()

row = df.iloc[i]
frame = row["frame_numbers"][0]
img_path = img_root / row["img_best"]
traj = row["trajectory"]



img = cv2.imread(str(img_path))
if img is None:
    st.error(f"Cannot load {img_path}")
else:
    draw_trajectory(img, traj)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)


st.subheader("Question")
st.write(row["question"])
st.write("**Answer:**", row["answer"])

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("NOT INCLUDE"):
        state.exclude.append(i)
        if i < total - 1:
            state.idx += 1
with c2:
    if st.button("INCLUDE"):
        if i < total - 1:
            state.idx += 1
with c3:
    if st.button("END"):
        if state.exclude:
            df.iloc[state.exclude].to_csv("excluded_cases.tsv", sep="\t", index=False)
            st.success("Saved excluded_cases.tsv")
        else:
            st.info("No exclusions.")
        st.stop()
