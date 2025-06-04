import json, sys, csv, cv2, math, re, random
from pathlib import Path
from PIL import Image

_SPLIT_RE = re.compile(r"VIRAT_S_(\d{5,6})")
scene_set = {"test": [0], "train": [20, 4000, 4010, 5000]}

def split_for(name: str) -> str:
    m = _SPLIT_RE.search(name)
    if not m:
        return "unsorted"
    sid = int(m.group(1)[:5])
    for split, ids in scene_set.items():
        if sid in ids:
            return split
    return "unsorted"

def smallest_bbox(ev, frame_idx):
    best, area = None, float("inf")
    for a in ev["trajectory"]:
        for bb in a.get("bbox_per_frame", []):
            if bb.get("frame") == frame_idx and {"mid_x", "mid_y", "bbox_width", "bbox_height"} <= bb.keys():
                ar = bb["bbox_width"] * bb["bbox_height"]
                if ar < area:
                    area = ar
                    x0 = bb["mid_x"] - bb["bbox_width"] / 2
                    y0 = bb["mid_y"] - bb["bbox_height"] / 2
                    best = [int(x0), int(y0), int(bb["bbox_width"]), int(bb["bbox_height"])]
    return best or []

def extract_centres(ev):
    pts = []
    for a in ev.get("trajectory", []):
        for bb in a.get("bbox_per_frame", []):
            if {"frame", "mid_x", "mid_y"} <= bb.keys():
                pts.append((bb["frame"], bb["mid_x"], bb["mid_y"]))
    pts.sort()
    return pts

def best_frame(selected, ev, W, H):
    cand, fb = [], []
    for a in ev.get("trajectory", []):
        for bb in a.get("bbox_per_frame", []):
            if bb.get("frame", -1) < selected:
                continue
            if not {"mid_x", "mid_y", "bbox_width", "bbox_height"} <= bb.keys():
                continue
            w, h = bb["bbox_width"], bb["bbox_height"]
            x0, y0 = bb["mid_x"] - w / 2, bb["mid_y"] - h / 2
            x1, y1 = x0 + w, y0 + h
            if 0 < x0 < x1 < W and 0 < y0 < y1 < H:
                cand.append((bb["frame"], w * h))
            else:
                fb.append((bb["frame"], w * h))
    pool = cand or fb
    return max(pool, key=lambda t: t[1])[0] if pool else selected

def question_answers(activity):
    qas = []
    # Vehicle turning
    if activity in ("vehicle_turning_left", "vehicle_turning_right"):
        turn_ans = "A" if activity == "vehicle_turning_left" else "B"
        qas.append((
            "What turn is the vehicle making? A Left Turn. B Right Turn. C None of the above.",
            turn_ans
        ))
        qas.append((
            "Is the vehicle making a U-Turn? A Yes. B No.",
            "B"
        ))
    # Vehicle moving
    elif activity in ("vehicle_moving_forward", "vehicle_moving_backward"):
        dir_ans = "A" if activity == "vehicle_moving_forward" else "B"
        qas.append((
            "In which direction is the vehicle moving? A Forward. B Backward. C None of the above.",
            dir_ans
        ))
        qas.append((
            "What turn is the vehicle making? A Left Turn. B Right Turn. C None of the above.",
            "C"
        ))
        qas.append((
            "Is the vehicle making a U-Turn? A Yes. B No.",
            "B"
        ))
    # Vehicle U-turn
    elif activity == "vehicle_u_turn":
        qas.append((
            "Is the vehicle making a U-Turn? A Yes. B No.",
            "A"
        ))
    # Person walking
    elif activity == "activity_walking":
        qas.append((
            "What is the person doing? A Entering the vehicle. B Exiting the vehicle. C None of the above.",
            "C"
        ))
    # Person entering/exiting
    elif activity in ("Entering", "Exiting"):
        pers_ans = "A" if activity == "Entering" else "B"
        qas.append((
            "What is the person doing? A Entering the vehicle. B Exiting the vehicle. C None of the above.",
            pers_ans
        ))
        qas.append((
            "In which direction is the vehicle moving? A Forward. B Backward. C None of the above.",
            "C"
        ))
    return qas



def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frm = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"frame {idx} missing in {video}")
    return cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

def save_plain(rgb, video, ev_id, idx, out_dir):
    fn = f"{Path(video).stem}__{ev_id}_frame{idx}.jpg"
    Image.fromarray(rgb).save(out_dir / fn)
    return fn


def process_event(key, ev, root):
    try:
        video = ev["source"].replace("/home/norm/data/", "/home/norm/workspace/") \
                         .replace("/annotations/", "/videos/")
        s_f_orig, e_f_orig = ev["start_frame"], ev["end_frame"]
        s_f = s_f_orig + 1
        e_f = max(s_f, e_f_orig - 1)
        m_f = (s_f + e_f) // 2

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {video}")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        b_f = best_frame(s_f, ev, W, H)

        out_dir = root / split_for(Path(video).name)
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = [("start", s_f), ("middle", m_f), ("end", e_f), ("best", b_f)]
        img_paths, frame_nums, obj_bboxes = [], [], []
        for _, idx in frames:
            rgb = read_frame(video, idx)
            fn = save_plain(rgb, video, ev["event_id"], idx, out_dir)
            img_paths.append(str((out_dir / fn).relative_to(root)))
            frame_nums.append(idx)
            obj_bboxes.append(smallest_bbox(ev, idx))

        centres = extract_centres(ev)
        xs = [x for _, x, _ in centres]
        ys = [y for _, _, y in centres]
        traj_bbox = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]

        # Q&A 
        qas = question_answers(ev.get("activity", ""))
        rows = []
        for question, answer in qas:
            rows.append([
                key,
                f"{question}\nAnswer with the option letter only",
                answer,
                *img_paths,
                "VIRAT",
                frame_nums,
                obj_bboxes,
                traj_bbox,
                json.dumps(ev.get("trajectory", []))
            ])
        return rows
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Warning: skipping event {key}: {e}")
        return []


def main():
    if len(sys.argv) != 2:
        print("Usage: python raw_extract_4frames.py <events.json>")
        sys.exit(1)

    data = json.load(open(sys.argv[1]))
    root = Path("/home/norm/workspace/VIRATVehicleDataset/raw")
    root.mkdir(exist_ok=True, parents=True)

    hdr = [
        "event_key", "question", "answer",
        "img_start", "img_mid", "img_end", "img_best",
        "source", "frame_numbers", "object_bbox", "trajectory_bbox",
        "trajectory"
    ]

    all_rows = []
    for k, v in data.items():
        all_rows.extend(process_event(k, v, root))

    with open(root / "data.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(hdr)
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows + images to {root}")

if __name__ == "__main__":
    main()
