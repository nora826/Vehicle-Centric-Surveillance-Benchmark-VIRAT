from __future__ import annotations
import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import re
import os



def parse_list(txt):
    return txt if isinstance(txt, list) else ast.literal_eval(txt)


def parse_json_blob(txt):
    if isinstance(txt, list):
        return txt
    try:
        return json.loads(txt.replace("\"\"", "\""))
    except json.JSONDecodeError:
        return ast.literal_eval(txt)


def distil_centres(traj_raw):
    centres: List[Tuple[int, float, float]] = []
    for actor in traj_raw:
        for bb in actor.get("bbox_per_frame", []):
            if {"frame", "mid_x", "mid_y"} <= bb.keys():
                centres.append((bb["frame"], bb["mid_x"], bb["mid_y"]))
    centres.sort(key=lambda t: t[0])
    return centres




def smallest_bbox(ev, frame_idx):
    best_box, best_area = None, float("inf")
    for actor in ev["trajectory_raw"]:
        for bb in actor.get("bbox_per_frame", []):
            if bb.get("frame") == frame_idx and {"mid_x", "mid_y", "bbox_width", "bbox_height"} <= bb.keys():
                area = bb["bbox_width"] * bb["bbox_height"]
                if area < best_area:
                    best_area = area
                    x0 = bb["mid_x"] - bb["bbox_width"] / 2
                    y0 = bb["mid_y"] - bb["bbox_height"] / 2
                    best_box = [int(x0), int(y0), int(bb["bbox_width"]), int(bb["bbox_height"])]
    return best_box


def trajectory_enclosing_bbox(traj_raw, *, margin=0):
    xmin, ymin = float("inf"), float("inf")

    xmax, ymax = float("-inf"), float("-inf")
    for actor in traj_raw:
        for bb in actor.get("bbox_per_frame", []):
            if {"mid_x", "mid_y", "bbox_width", "bbox_height"} <= bb.keys():
                x0 = bb["mid_x"] - bb["bbox_width"] / 2
                y0 = bb["mid_y"] - bb["bbox_height"] / 2
                x1, y1 = x0 + bb["bbox_width"], y0 + bb["bbox_height"]
                xmin, ymin = min(xmin, x0), min(ymin, y0)
                xmax, ymax = max(xmax, x1), max(ymax, y1)
    if xmin == float("inf"):
        return None
    xmin, ymin = int(max(0, xmin - margin)), int(max(0, ymin - margin))
    xmax, ymax = int(xmax + margin), int(ymax + margin)
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def draw_bbox(img, box, colour=(0, 255, 0), thick=2):
    if not box or len(box) != 4:
        return
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, thick)


def draw_trajectory(img, centres, *, arrow=False):
    if len(centres) < 2:
        return
    pts = [(int(x), int(y)) for _, x, y in centres]
    for p, q in zip(pts, pts[1:]):
        cv2.line(img, p, q, (0, 0, 255), 3)
    if not arrow:
        return

    # draw arrow head
    dirs = []
    for p, q in reversed(list(zip(pts, pts[1:]))):
        dx, dy = q[0] - p[0], q[1] - p[1]
        l = math.hypot(dx, dy)
        if l > 2:
            dirs.append((dx / l, dy / l))
            if len(dirs) == 6:
                break
    if not dirs:
        return
    ux = sum(d[0] for d in dirs) / len(dirs)
    uy = sum(d[1] for d in dirs) / len(dirs)
    norm = math.hypot(ux, uy)
    if norm < 1e-6:
        return
    ux, uy = ux / norm, uy / norm
    ang = math.radians(30)
    cos_a, sin_a = math.cos(ang), math.sin(ang)
    lx, ly = ux * cos_a - uy * sin_a, ux * sin_a + uy * cos_a
    rx, ry = ux * cos_a + uy * sin_a, -ux * sin_a + uy * cos_a
    tip = pts[-1]
    L = (int(tip[0] - 20 * lx), int(tip[1] - 20 * ly))
    R = (int(tip[0] - 20 * rx), int(tip[1] - 20 * ry))
    cv2.line(img, tip, L, (0, 0, 255), 3)
    cv2.line(img, tip, R, (0, 0, 255), 3)


# image generation

def generate_image(fi, *, bbox, trajectory, arrow, traj_box, blur, crop, show_traj_box=True):

    img = cv2.imread(fi["path"])
    
    if crop and traj_box is not None:
        x0, y0, w, h = traj_box
        cropped = img[y0 : y0 + h, x0 : x0 + w].copy()

        
        if bbox:
            orig_box = smallest_bbox(fi, fi["frame"])
            if orig_box:
                shifted_box = [orig_box[0] - x0, orig_box[1] - y0, orig_box[2], orig_box[3]]
                draw_bbox(cropped, shifted_box)

        if trajectory:
            orig_centres = fi["traj"]
            shifted_centres = [(f, x - x0, y - y0) for (f, x, y) in orig_centres]
            draw_trajectory(cropped, shifted_centres, arrow=arrow)

        if show_traj_box:
            draw_bbox(cropped, [0, 0, w, h], colour=(0, 255, 255), thick=2)

        return cropped

   
    if bbox:
        draw_bbox(img, smallest_bbox(fi, fi["frame"]))

    if trajectory:
        draw_trajectory(img, fi["traj"], arrow=arrow)

    if show_traj_box and traj_box is not None:
        draw_bbox(img, traj_box, colour=(0, 255, 255), thick=2)

    if blur and traj_box is not None:
        x, y, w, h = traj_box
        blurred = cv2.GaussianBlur(img, (41, 41), 0)
        blurred[y : y + h, x : x + w] = img[y : y + h, x : x + w]
        img = blurred

    return img




base_path = Path("/home/norm/workspace/VIRATVehicleDataset/raw")
data_path = base_path / "trajectories_V4.tsv"
img_root = base_path

@st.cache_resource(show_spinner=False)
def load_dataframe(path):
    df = pd.read_csv(path, sep="\t")
    for c in ("frame_numbers", "object_bbox", "trajectory_bbox"):
        df[c] = df[c].apply(parse_list)
    df["trajectory_raw"] = df["trajectory"].apply(parse_json_blob)
    df["trajectory"] = df["trajectory_raw"].apply(distil_centres)
    return df


df = load_dataframe(data_path)





include_none = st.sidebar.checkbox("Include 'None of the above'", value=True)

all_questions = df["question"].unique().tolist()

question_checks: dict[str, bool] = {}
st.sidebar.header("Which questions do you want to include?")
for q in all_questions:
    question_checks[q] = st.sidebar.checkbox(q, value=True)



selected_questions = [q for q, checked in question_checks.items() if checked]
df_filtered = df[df["question"].isin(selected_questions)].reset_index(drop=True) # keep only selected questions

# if we are not including none of the above samples, remove that option from the question
if not include_none:
    df_filtered = df_filtered[df_filtered["answer"] != "C"].reset_index(drop=True)

N_filtered = len(df_filtered)






if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

N_full = len(df)
row = df.iloc[st.session_state.row_idx]




st.sidebar.header("Display options")
mode = st.sidebar.radio("Mode", ("1x1", "3x1"))
show_bbox    = st.sidebar.checkbox("Show main-actor bbox")
show_traj    = st.sidebar.checkbox("Show trajectory")
show_arrow   = st.sidebar.checkbox("Arrow head", disabled=not show_traj)


for key in ("show_blur", "show_crop", "show_trajbox"):
    if key not in st.session_state:
        st.session_state[key] = False


# only one of these 3 can be true at the same time
show_blur = st.sidebar.checkbox(
    "Blur non-relevant area",
    key="show_blur",
    disabled=(st.session_state.show_crop or st.session_state.show_trajbox),
)

show_crop = st.sidebar.checkbox(
    "Crop to trajectory bbox",
    key="show_crop",
    disabled=(st.session_state.show_blur or st.session_state.show_trajbox),
)

show_trajbox = st.sidebar.checkbox(
    "Trajectory enclosing bbox",
    key="show_trajbox",
    disabled=(st.session_state.show_blur or st.session_state.show_crop),
)


if st.session_state.show_blur and (st.session_state.show_crop or st.session_state.show_trajbox):
    st.session_state.show_crop = False
    st.session_state.show_trajbox = False
    st.experimental_rerun()
if st.session_state.show_crop and (st.session_state.show_blur or st.session_state.show_trajbox):
    st.session_state.show_blur = False
    st.session_state.show_trajbox = False
    st.experimental_rerun()
if st.session_state.show_trajbox and (st.session_state.show_blur or st.session_state.show_crop):
    st.session_state.show_blur = False
    st.session_state.show_crop = False
    st.experimental_rerun()


show_blur = st.session_state.show_blur
show_crop = st.session_state.show_crop
show_trajbox = st.session_state.show_trajbox

st.sidebar.header("Export options")
save_traj      = st.sidebar.checkbox("Save trajectory")
save_bbox      = st.sidebar.checkbox("Save bbox")
save_traj_bbox = st.sidebar.checkbox("Save trajectory bbox")

colA, colB, colC = st.columns(3)
with colA:
    gen_btn = st.button("Generate Preview", type="primary")
with colB:
    nxt_btn = st.button("Next", type="secondary")
with colC:
    export_btn = st.button("Generate dataset with this configuration", type="primary")




def current_config_dict():
    return dict(
        mode=mode,
        bbox=show_bbox,
        trajectory=show_traj,
        arrow=show_arrow,
        traj_bbox=show_trajbox,
        blur_area=show_blur,
        crop_image=show_crop,
        save_traj=save_traj,
        save_bbox=save_bbox,
        save_traj_bbox=save_traj_bbox,
        include_none=include_none,
        questions_included=selected_questions,
    )


def unpack(r):
    names = ["img_start", "img_mid", "img_end", "img_best"]
    frames = r["frame_numbers"]
    bboxes = r["object_bbox"]
    d = {}
    for i, n in enumerate(names):
        d[n] = dict(frame=frames[i],obj_bbox=bboxes[i] if i < len(bboxes) else [], path=str(img_root / r[n]), trajectory_raw=r["trajectory_raw"], traj=r["trajectory"],)
    return d


_FRAME_RE = re.compile(r"_frame\d+$")

def clean_name(path, *, suffix=None):
    p = Path(path)
    stem = _FRAME_RE.sub("", p.stem)
    return f"{stem}{suffix or ''}{p.suffix}"


def make_preview(row_idx):
 
    row = df.iloc[row_idx]
    info = unpack(row)
    traj_box = trajectory_enclosing_bbox(row["trajectory_raw"]) if (show_trajbox or show_blur or show_crop) else None

    def join():
        render = generate_image
        if mode == "1x1":
            return render(
                info["img_best"],
                bbox=show_bbox,
                trajectory=show_traj,
                arrow=show_arrow,
                traj_box=traj_box,
                blur=show_blur,
                crop=show_crop,
                show_traj_box=show_trajbox,
            )
        return np.hstack([
            render(info["img_start"], bbox=show_bbox, trajectory=show_traj, arrow=show_arrow,
                   traj_box=traj_box, blur=show_blur, crop=show_crop, show_traj_box=show_trajbox),
            render(info["img_mid"],   bbox=show_bbox, trajectory=show_traj, arrow=show_arrow,
                   traj_box=traj_box, blur=show_blur, crop=show_crop, show_traj_box=show_trajbox),
            render(info["img_end"],   bbox=show_bbox, trajectory=show_traj, arrow=show_arrow,
                   traj_box=traj_box, blur=show_blur, crop=show_crop, show_traj_box=show_trajbox),
        ])

    return join()



if nxt_btn:
    st.session_state.row_idx = (st.session_state.row_idx + 1) % N_full
    st.rerun()

if gen_btn:
    preview_img = make_preview(st.session_state.row_idx)
    st.image(
        cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )





def export_dataset(filtered_df, config):
   
    base_dir = Path.cwd()
    k = 1
    while (base_dir / f"configuration_{k}").exists():
        k += 1
    out_dir   = base_dir / f"configuration_{k}"
    train_dir = out_dir / "train"
    test_dir  = out_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()

    tsv_rows = []
    prog = st.progress(0.0, text="Exporting dataset …")
    total = len(filtered_df)

    for i, r in enumerate(filtered_df.itertuples(index=False)):
        row_dict = r._asdict()
        info = unpack(row_dict)

        traj_box = trajectory_enclosing_bbox(row_dict["trajectory_raw"]) if (
            config["traj_bbox"] or config["blur_area"] or config["crop_image"]) else None

        def render_img(fi):
            return generate_image(
                fi,
                bbox=config["bbox"],
                trajectory=config["trajectory"],
                arrow=config["arrow"],
                traj_box=traj_box,
                blur=config["blur_area"],
                crop=config["crop_image"],
                show_traj_box=config["traj_bbox"],
            )

        if config["mode"] == "1x1":
            rendered = render_img(info["img_best"])
            rel_name = clean_name(info["img_best"]["path"])
            frame_nums = [row_dict["frame_numbers"][0]]
        else:
            rendered = np.hstack([
                render_img(info["img_start"]),
                render_img(info["img_mid"]),
                render_img(info["img_end"]),
            ])
            rel_name = clean_name(info["img_start"]["path"], suffix="_strip")
            frame_nums = row_dict["frame_numbers"][:3]

        subset_dir = train_dir if "train/" in row_dict["img_start"] else test_dir
        cv2.imwrite(str(subset_dir / rel_name), rendered)

        question_text = row_dict["question"]
        if not config["include_none"]:

            question_text = re.sub(r"\s*C\s*None of the above\.\s*", " ", question_text)
            question_text = re.sub(r"\s{2,}", " ", question_text).strip()

        final_question = question_text.strip() + f"\n{row_dict['answer']}"

        entry={
            "event_key":   row_dict["event_key"],
            "question":    final_question,
            "img":         f"{subset_dir.name}/{rel_name}",
            "source":      row_dict["source"],
            "frame_number": frame_nums,
        }

        if config["save_bbox"]:
            entry["object_bbox"] = (
                row_dict["object_bbox"][0]
                if config["mode"] == "1x1"
                else row_dict["object_bbox"][:3]
            )
        if config["save_traj_bbox"]:
            entry["trajectory_bbox"] = traj_box
        if config["save_traj"]:
            entry["trajectory"] = row_dict["trajectory"]

        tsv_rows.append(entry)
        prog.progress((i + 1) / total)

    pd.DataFrame(tsv_rows).to_csv(out_dir / "data.tsv", sep="\t", index=False)

    with open(out_dir / "configuration_info.txt", "w") as fh:
        for k, v in config.items():
            fh.write(f"{k}: {v}\n")

    prog.empty()
    st.success(f"Dataset exported to {out_dir}")


if export_btn:
    export_dataset(df_filtered, current_config_dict())





st.title("VIRAT Vehicle Event Explorer")
st.subheader(f"Event: {row.event_key}")
st.write(row.question)
st.caption(f"Row {st.session_state.row_idx + 1} / {N_full}")
