from __future__ import annotations
import sys
import csv
import re
import traceback
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
)
from transformers.generation import GenerationMixin

DEFAULT_MODEL_DIR  = "/home/norm/workspace/TFG/Testing/models/internvl2-2b"
DEFAULT_MAX_LENGTH = 50
DEFAULT_BATCH_SIZE = 1

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers multiple-choice questions about "
    "surveillance images. Your answer must be exactly one of the option letters."
)


def strip_answer_suffix(text):
    out = []
    for ln in text.splitlines():
        if ln.strip().lower().startswith("answer with"):
            break
        out.append(ln)
    return "\n".join(out).rstrip()


def _load_processor(model_dir):
    try:
        return AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        return AutoFeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)


def _ensure_generation_mixin(cls):
    if not issubclass(cls, GenerationMixin):
        cls.__bases__ += (GenerationMixin,)
    return cls


def caption_batch(
    images:  List[Image.Image],
    prompts: List[str],
    proc, tok, model, device, max_len: int,
):
    merged = [f"{SYSTEM_PROMPT}\n{p}" for p in prompts]
    pixel_vals = proc(images=images, return_tensors="pt").pixel_values.to(
        device=device, dtype=model.dtype
    )
    gen_cfg = dict(max_new_tokens=max_len, do_sample=False)
    return model.batch_chat(
        tok, pixel_vals, merged, gen_cfg, [1] * len(images)
    )


def main(path):
    data_dir = Path(path).expanduser().resolve()
    tsv_in   = data_dir / "data.tsv"
    if not tsv_in.is_file():
        sys.exit(f"[✘] data.tsv not found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc   = _load_processor(DEFAULT_MODEL_DIR)
    tok    = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, trust_remote_code=True)
    tmp    = AutoModelForCausalLM.from_pretrained(
                 DEFAULT_MODEL_DIR,
                 trust_remote_code=True,
                 torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
                 low_cpu_mem_usage=True,
             )
    Model  = _ensure_generation_mixin(tmp.__class__)
    model  = Model.from_pretrained(
                 DEFAULT_MODEL_DIR,
                 trust_remote_code=True,
                 torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
                 low_cpu_mem_usage=True,
             ).to(device).eval()
    del tmp

    out_rows = []
    batch_event_keys = []
    batch_imgs       = []
    batch_prompts    = []
    batch_real_ans   = []
    batch_paths      = []

    with open(tsv_in, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in tqdm(reader, desc="Inference"):
            event_key = row.get("event_key", "").strip()
            rel       = row.get("img",       "").strip()
            raw_q     = row.get("question",  "").strip()

            gt = None
            for ln in reversed(raw_q.splitlines()):
                if re.fullmatch(r"[A-Za-z]", ln.strip()):
                    gt = ln.strip().upper()
                    break
            real_answer = gt or ""

            qtext = strip_answer_suffix(raw_q)

            img_path = Path(rel) if Path(rel).is_absolute() else data_dir / rel
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[!] {img_path}: {e}", file=sys.stderr)
                out_rows.append((event_key, rel, qtext, real_answer, ""))
                continue

            batch_event_keys.append(event_key)
            batch_paths.append(rel)
            batch_prompts.append(qtext)
            batch_real_ans.append(real_answer)
            batch_imgs.append(img)

            if len(batch_imgs) >= DEFAULT_BATCH_SIZE:
                try:
                    outs = caption_batch(
                        batch_imgs, batch_prompts,
                        proc, tok, model, device, DEFAULT_MAX_LENGTH
                    )
                except Exception:
                    traceback.print_exc()
                    outs = [""] * len(batch_imgs)

                out_rows.extend(zip(
                    batch_event_keys,
                    batch_paths,
                    batch_prompts,
                    batch_real_ans,
                    outs
                ))
                batch_event_keys.clear()
                batch_paths.clear()
                batch_prompts.clear()
                batch_real_ans.clear()
                batch_imgs.clear()

        
        if batch_imgs:
            try:
                outs = caption_batch(
                    batch_imgs, batch_prompts,
                    proc, tok, model, device, DEFAULT_MAX_LENGTH
                )
            except Exception:
                traceback.print_exc()
                outs = [""] * len(batch_imgs)

            out_rows.extend(zip(
                batch_event_keys,
                batch_paths,
                batch_prompts,
                batch_real_ans,
                outs
            ))

    tsv_out = data_dir / "internVL_results.tsv"
    with open(tsv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["event_key", "img", "question", "real_answer", "model_answer"])
        w.writerows(out_rows)

    print("Finished!!!!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/data_directory")
        sys.exit(1)
    main(sys.argv[1])
