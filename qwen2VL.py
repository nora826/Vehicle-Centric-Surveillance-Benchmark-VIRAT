import argparse, io, sys, time, gc
from pathlib import Path
from typing import List, Dict

import cv2, pandas as pd, numpy as np
from PIL import Image
import torch
import re


MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"   
GPU_LAYERS  = -1                             
N_CTX       = 32                           
MAX_EDGE    = 1024                         

PANDA_LLM_PATH = "/home/norm/workspace/Panda-LLM"   
sys.path.append(PANDA_LLM_PATH)

from modules import shared
from webui   import load_model


def free_cuda_memory():
    torch.cuda.empty_cache()
    gc.collect()


def preprocess_image(img, max_edge=MAX_EDGE, pad_to_square=False):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if pad_to_square and h != w:
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


def convert_to_string(x):
    import io
    if isinstance(x, list):
        return convert_to_string(x[-1])
    if isinstance(x, tuple):
        return " ".join(map(convert_to_string, x))
    if isinstance(x, dict):
        return str({k: convert_to_string(v) for k, v in x.items()})
    if isinstance(x, io.BytesIO):
        return "<BytesIO>"
    return str(x)


def extract_letter(raw, valid_letters):
    raw_low = raw.lower()
  
    cls = "".join(valid_letters)
    m = re.search(rf"\b([{cls}])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
 
    verbal_map = {
        # forward/backward/none
        "forward":  "A",
        "backward": "B",
        "none":     "C",
        # left/right
        "left":     "A",
        "right":    "B",
        # yes/no
        "yes":      "A",
        "no":       "B",
        # entering/exiting
        "entering": "A",
        "exiting":  "B",
        # straight
        "straight": "C",
    }
    for word, letter in verbal_map.items():
        if word in raw_low and letter in valid_letters:
            return letter

    return "?"


def run_model(image, prompt):
    image = Image.fromarray(image[:, :, ::-1]) 
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    chatbot = [((buf,), None)]
    _, chatbot = shared.model.append_user_input(prompt, chatbot)
    params = dict(
        top_k=-1,
        top_p=0.8,
        temperature=0.5,
        enable_postprocessing=True,
        system_prompt="You are a helpful assistant that answers multiple-choice questions about surveillance images. Your answer must be exactly one of the option letters.",
        max_new_tokens=12,
    )
    answer = ""
    try:
        for out in shared.model.predict(chatbot, params):
            answer = out
    except torch.cuda.OutOfMemoryError:
        answer = "CUDA OOM"
        free_cuda_memory()
    except Exception as e:
        answer = f"Error: {e}"
    return convert_to_string(answer)


def cli():
    ap = argparse.ArgumentParser(description="Benchmark Qwen-VL on VIRAT slim TSV")
    ap.add_argument("dataset_dir", help="directory containing data.tsv, train/, test/")
    return ap.parse_args()


def main():
    args = cli()
    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    slim_tsv = ds_dir / "data.tsv"
    results_tsv = ds_dir / "qwen_2B_results.tsv"

    print("Loading model …")
    load_model(MODEL_NAME, GPU_LAYERS, N_CTX, lora_path="", load_in_8bit=False)
    print("Model ready.")
    free_cuda_memory()

    df = pd.read_csv(slim_tsv, sep="\t")
    records = []

    for i, row in df.iterrows():
        ev = row.event_key
        qraw = row.question
    
        lines = qraw.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) > 1:
            gt_letter = lines[-1].strip()[0].upper()
            q_text = "\n".join(lines[:-1])
        else:
            q_text, gt_letter = qraw, "?"
  
        opts = re.findall(r"^([A-Z])\)", q_text, re.MULTILINE)
        valid = [o.upper() for o in opts] or ["A","B","C"]

        img_path = ds_dir / row.img
        img = cv2.imread(str(img_path))
        if img is not None:
            img = preprocess_image(img)
            free_cuda_memory()
            t0 = time.time()
            raw_out = run_model(img, q_text)
            dt = time.time() - t0
            model_ans = extract_letter(raw_out, valid)
        else:
            dt = None
            model_ans = "?"
        print(f"[{i+1}/{len(df)}] {ev} → raw: {raw_out!r} ans: {model_ans} (gt={gt_letter})")
        records.append({
            "event_key": ev,
            "question": q_text,
            "real_answer": gt_letter,
            "model_answer": model_ans,
            "processing_s": dt,
        })

    pd.DataFrame(records).to_csv(results_tsv, sep="\t", index=False,
                                 columns=["event_key","question","real_answer","model_answer","processing_s"])
    print("Finished!!!!")


if __name__ == "__main__":
    main()
