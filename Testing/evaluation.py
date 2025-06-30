#!/usr/bin/env python3
import os
import sys
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_responses(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    required = ["event_key", "question", "real_answer", "model_answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["question"] = (
        df["question"]
        .str.replace("Answer with the option letter only", "", regex=False)
        .str.strip()
    )
    df["question"] = df["question"].str.extract(r'^(.*?\?)')[0].fillna(df["question"])


    df["real_answer"] = (
        df["real_answer"]
        .str.strip()
        .str.upper()
        .str.extract(r'^([A-Z])')[0]
    )
    df["model_answer"] = (
        df["model_answer"]
        .str.strip()
        .str.upper()
        .str.extract(r'^([A-Z])')[0]
    )


    df["processing_s"] = pd.to_numeric(df.get("processing_s", pd.NA), errors="coerce")
    return df

def evaluate_metrics(df):
    stats = []
    for question, group in df.groupby("question", sort=False):
        if not question:
            continue
        samples = len(group)
        acc = (group["model_answer"] == group["real_answer"]).mean()
        avg_rt = group["processing_s"].dropna().astype(float).mean()
        stats.append({
            "question": question,
            "samples": samples,
            "accuracy": acc,
            "avg_runtime_s": avg_rt
        })
    return pd.DataFrame(stats)

def plot_confusion(df, question, save_path=None):
    subdf = df[df["question"].str.contains(question, case=False, na=False)]
    if subdf.empty:
        print(f"[Warning] No samples for question: {question}")
        return

    # Only letters A/B/C will remain
    labels = sorted(set(subdf["real_answer"]) | set(subdf["model_answer"]))
    cm = confusion_matrix(subdf["real_answer"], subdf["model_answer"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=True, values_format="d")
    ax.set_title(question)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def evaluation_metrics(config, model, data_path, stats_df):
    question_map = {
        "What Turn": "what turn",
        "U-Turn": "u-turn",
        "Direction": "direction",
        "Person Action": "person"
    }
    row = {"configuration": config, "model": model}
    for label, kw in question_map.items():
        subset = stats_df[stats_df["question"].str.lower().str.contains(kw, na=False)]
        row[label] = subset["accuracy"].mean() if not subset.empty else pd.NA
    row["average accuracy"] = stats_df["accuracy"].mean()
    row["average runtime"] = stats_df["avg_runtime_s"].mean()

    csv_file = "results/evaluation_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["configuration", "model"] + list(question_map) + ["average accuracy", "average runtime"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return True

def main(path):
    model  = re.sub(r"(_results\.tsv)$", "", os.path.basename(path))
    config = os.path.basename(os.path.dirname(path))
    df     = load_responses(path)
    stats  = evaluate_metrics(df)

    if evaluation_metrics(config, model, path, stats):
        print("Wrote evaluation metrics.")

    plots_dir = "/home/norm/workspace/TFG/Testing/results"
    question_map = {
        "What Turn": "what turn",
        "U-Turn": "u-turn",
        "Direction": "direction",
        "Person Action": "person"
    }
    for label, kw in question_map.items():
        out_png = f"{plots_dir}/{config}/{label}_{model}.png"
        plot_confusion(df, kw, save_path=out_png)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py /path/to/results.tsv")
        sys.exit(1)
    main(sys.argv[1])
