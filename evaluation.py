from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,f1_score,confusion_matrix, ConfusionMatrixDisplay)


QUESTION_FAMILIES: dict[str, list[str]] = {
    "What-Turn":   ["what turn"],
    "U-Turn":      ["u-turn"],
    "Direction":   ["direction"],
    "Enter/Exit":  ["person"],     
}

METRIC_NAMES = ["accuracy", "macroF1"]          


def load_responses(tsv_path) :
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    need = {"event_key", "question", "real_answer", "model_answer"}
    if not need.issubset(df.columns):
        raise KeyError(f"{tsv_path}: missing {need - set(df.columns)}")

    df["question"] = (
        df["question"]
        .str.replace("Answer with the option letter only", "", regex=False)
        .str.extract(r"^(.*?\?)")[0]
        .fillna(df["question"])
        .str.strip()
    )
    for col in ("real_answer", "model_answer"):
        df[col] = df[col].str.upper().str.extract(r"^([A-Z])")[0].fillna("?")

    df["processing_s"] = pd.to_numeric(df.get("processing_s"), errors="coerce")
    return df


def compute_metrics(y_true, y_pred):
    labels = sorted({*y_true, *y_pred})
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macroF1":  f1_score(y_true, y_pred,
                             average="macro", labels=labels, zero_division=0),
    }


def save_confusion_png(df,labels,title, out_png) :
    cm = confusion_matrix(df["real_answer"], df["model_answer"], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def process_one_file(tsv):
    cfg   = tsv.parent.name                         
    model = re.sub(r"_results\.tsv$", "", tsv.name) 

    df = load_responses(tsv)
    out = {"config": cfg, "model": model}

    out.update(compute_metrics(df["real_answer"], df["model_answer"]))
    out["avg_runtime"] = df["processing_s"].mean()

    for fam, kws in QUESTION_FAMILIES.items():
        mask = df["question"].str.lower().apply(
            lambda q: any(kw in q for kw in kws)
        )
        if not mask.any():
            continue

        fam_stats = compute_metrics(df.loc[mask, "real_answer"],
                                    df.loc[mask, "model_answer"])
        for k in METRIC_NAMES:
            out[f"{fam}_{k}"] = fam_stats[k]

        labels = sorted({*df.loc[mask, "real_answer"],
                         *df.loc[mask, "model_answer"]})
        png_out = (tsv.parent.parent / "plots" / cfg / model /
                   f"{fam.replace(' ','_')}.png")
        save_confusion_png(df.loc[mask], labels, fam, png_out)

    return out



def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dir containing <config>/*_results.tsv files " "or a single *_results.tsv file")
    return ap.parse_args()


def main():
    args = cli()
    root = Path(args.root).expanduser().resolve()

    tsv_files = [root] if root.is_file() else list(root.rglob("*_results.tsv"))
    if not tsv_files:
        sys.exit("No *_results.tsv files found.")

    rows: list[Dict[str, str | float]] = []
    for tsv in tsv_files:
        try:
            rows.append(process_one_file(tsv))
            print(f"processed {tsv}")
        except Exception as e:
            print(f"[!] {tsv}: {e}", file=sys.stderr)

    out_csv = (root / "evaluation_metrics.csv"
               if root.is_dir() else root.with_suffix(".csv"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved in {out_csv}")


if __name__ == "__main__":
    main()
