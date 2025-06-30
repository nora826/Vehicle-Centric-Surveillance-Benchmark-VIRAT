#!/usr/bin/env python3
import sys, json, csv, cv2, re
from pathlib import Path
from PIL import Image

def smallest_bbox(ev, frame):
    best, area = [], float("inf")
    for a in ev["trajectory"]:
        for bb in a.get("bbox_per_frame", []):
            if bb.get("frame")==frame and {"mid_x","mid_y","bbox_width","bbox_height"} <= bb:
                ar = bb["bbox_width"]*bb["bbox_height"]
                if ar<area:
                    area=ar
                    x0,y0=bb["mid_x"]-bb["bbox_width"]/2, bb["mid_y"]-bb["bbox_height"]/2
                    best=[int(x0),int(y0),int(bb["bbox_width"]),int(bb["bbox_height"])]
    return best

def extract_centres(ev):
    pts = [(bb["frame"],bb["mid_x"],bb["mid_y"])
           for a in ev["trajectory"]
           for bb in a.get("bbox_per_frame",[])
           if {"frame","mid_x","mid_y"}<=bb]
    return sorted(pts)

def best_frame(start, ev):
    cand = [bb["frame"]
            for a in ev["trajectory"]
            for bb in a.get("bbox_per_frame",[])
            if bb.get("frame",−1)>=start and {"mid_x","mid_y","bbox_width","bbox_height"}<=bb]
    return min(cand, default=start)

def find_video(name):
    base = Path("/home/norm/workspace/TFG/Question-Dataset/VIRAT-VIDEOS/videos_original")
    alt  = Path("//home/norm/workspace/VIRAT")
    stem = Path(name).stem
    for d in (base, alt):
        for ext in ("", ".mp4"):
            p = d/ (stem+ext)
            if p.is_file(): return str(p)
    return None

def question_answers(act):
    Q={}
    if act.startswith("vehicle_turning"):
        Q["What turn is the vehicle making? A Left Turn. B Right Turn. C None."] = "A" if "left" in act else "B"
        Q["Is the vehicle making a U-Turn? A Yes. B No."] = "B"
    elif act.startswith("vehicle_moving"):
        Q["In which direction is the vehicle moving? A Forward. B Backward. C None."] = "A" if "forward" in act else "B"
        Q["What turn is the vehicle making? A Left Turn. B Right Turn. C None."]="C"
        Q["Is the vehicle making a U-Turn? A Yes. B No."]="B"
    elif act=="vehicle_u_turn":
        Q["Is the vehicle making a U-Turn? A Yes. B No."]="A"
    elif act=="activity_walking":
        Q["What is the person doing? A Getting into. B Getting out. C None."]="C"
    elif "Person getting" in act:
        Q["What is the person doing? A Getting into. B Getting out. C None."] = "A" if "into" in act else "B"
        Q["In which direction is the vehicle moving? A Forward. B Backward. C None."]="C"
    return list(Q.items())

def read_frame(video, idx):
    cap=cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok,frm=cap.read(); cap.release()
    if not ok: raise RuntimeError(f"frame {idx} missing")
    return cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

def save_img(rgb, video, eid, idx, out):
    fn=f"{Path(video).stem}__{eid}_frame{idx}.jpg"
    Image.fromarray(rgb).save(out/fn)
    return fn

def process_event(key, ev, imgs):
    vid=find_video(Path(ev["source"]).name)
    if not vid: return []
    s,e = ev["start_frame"]+1, ev["end_frame"]-1
    m=(s+max(s,e))/2
    b=best_frame(s,ev)
    frames=[("start",s),("middle",int(m)),("end",e),("best",b)]
    imgs.mkdir(parents=True,exist_ok=True)
    paths,nums,boxes = [],[],[]
    for _,f in frames:
        rgb=read_frame(vid,f)
        fn=save_img(rgb,vid,ev["event_id"],f,imgs)
        paths.append(str((imgs/fn).relative_to(imgs.parent)))
        nums.append(f)
        boxes.append(smallest_bbox(ev,f) or [])
    xs=[x for _,x,_ in extract_centres(ev)]; ys=[y for _,_,y in extract_centres(ev)]
    traj_box=[int(min(xs)),int(min(ys)),int(max(xs)-min(xs)),int(max(ys)-min(ys))]
    rows=[]
    for q,a in question_answers(ev.get("activity","")):
        rows.append([
            key,
            f"{q}\nAnswer with the option letter only",
            a,
            *paths,
            ev.get("dataset",""),
            nums,
            boxes,
            traj_box,
            ev.get("dataset",""),
            json.dumps(ev.get("trajectory",[]))
        ])
    return rows

def main():
    if len(sys.argv)!=2:
        print("Usage: raw_extract_4frames.py <events.json>"); sys.exit(1)
    data=json.load(open(sys.argv[1]))
    root=Path("/home/norm/workspace/TFG/Question-Dataset/raw-dataset")
    imgs=root/"images"
    hdr=["event_key","question","answer",
         "img_start","img_mid","img_end","img_best",
         "source","frame_numbers","object_bbox","trajectory_bbox",
         "dataset","trajectory"]
    all_rows=[]
    for k,v in data.items():
        all_rows+=process_event(k,v,imgs)
    with open(root/"data.tsv","w",newline="") as f:
        w=csv.writer(f,delimiter="\t")
        w.writerow(hdr)
        w.writerows(all_rows)
    print(f"Written {len(all_rows)} rows to {root}")

if __name__=="__main__":
    main()
