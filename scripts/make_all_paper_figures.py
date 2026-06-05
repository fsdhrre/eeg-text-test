#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate one folder containing all paper-ready figures.

This script gathers the already-computed experiment outputs and creates a
compact set of figures that can be directly used in the paper:

1. Main pipeline diagram.
2. Main metric comparison bar chart.
3. Ablation heatmap.
4. Relative improvement over the high-only baseline.
5. t-SNE visualizations copied from the classification analysis.
6. Per-class classification accuracy bar chart.
7. Qualitative evidence/caption examples.
8. Paper metric table rendered as an image.

All outputs are saved to one directory so the paper-writing stage is less
messy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst / src.name)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_pipeline(out_dir: Path) -> None:
    stages = [
        ("Raw EEG", "Test EEG signal"),
        ("EEGPT Encoder", "Frozen pretrained backbone"),
        ("Adapter + Heads", "Learn EEG semantic features"),
        ("Semantic Retrieval", "High evidence + reliable low evidence"),
        ("Evidence Decision", "Filter unreliable cues"),
        ("Structured Prompt", "Object + scene + visual attributes"),
        ("LLM Generator", "One-sentence visual caption"),
    ]

    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")

    x_positions = np.linspace(0.04, 0.96, len(stages))
    colors = ["#4C78A8", "#59A14F", "#59A14F", "#F28E2B", "#F28E2B", "#B07AA1", "#E15759"]

    for i, ((title, subtitle), x, color) in enumerate(zip(stages, x_positions, colors)):
        box = plt.Rectangle((x - 0.065, 0.38), 0.13, 0.32, fc=color, ec="white", lw=1.5, alpha=0.95)
        ax.add_patch(box)
        ax.text(x, 0.59, title, ha="center", va="center", color="white", weight="bold", fontsize=10)
        ax.text(x, 0.47, subtitle, ha="center", va="center", color="white", fontsize=7.5, wrap=True)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.075, 0.54),
                xytext=(x + 0.075, 0.54),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
            )

    ax.text(
        0.5,
        0.16,
        "Core idea: convert EEG into retrievable semantic evidence, then let the LLM generate a conservative caption from reliable cues.",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    savefig(fig, out_dir, "01_method_pipeline")


def draw_metric_table(out_dir: Path, table_csv: Path) -> None:
    if not table_csv.exists():
        return
    df = pd.read_csv(table_csv)
    rename_map = {
        "method": "Method",
        "object_accuracy": "ObjectAcc",
        "token_f1": "TokenF1",
        "bleu1_sentence_avg": "BLEU1",
        "bleu2_sentence_avg": "BLEU2",
        "rouge_l": "ROUGE-L",
        "meteor": "METEOR",
        "cider": "CIDEr",
        "evidence_faithfulness": "Faith",
        "image_text_clipscore": "Image-CLIP",
    }
    df = df.rename(columns=rename_map)
    preferred = ["Method", "ObjectAcc", "Image-CLIP", "TokenF1", "BLEU1", "BLEU2", "ROUGE-L", "METEOR", "CIDEr", "Faith"]
    columns = [c for c in preferred if c in df.columns]
    if not columns:
        columns = df.columns[: min(len(df.columns), 8)].tolist()
    df = df[columns].copy()
    for col in df.columns:
        if col != "Method":
            df[col] = df[col].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 1.25), 1.3 + len(df) * 0.55))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2F4B7C")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6FA")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D0D5DD")
    ax.set_title("Main Captioning Results", pad=16, weight="bold")
    savefig(fig, out_dir, "02_main_result_table")


def draw_per_class_accuracy(out_dir: Path, csv_path: Path, top_n: int = 28) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if not {"true_label", "accuracy", "count"}.issubset(df.columns):
        return
    df = df.sort_values("accuracy", ascending=False).head(top_n)

    summary_path = csv_path.with_name(csv_path.name.replace("_per_class_accuracy.csv", "_summary.json"))
    summary_text = ""
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary_text = f"Full test classifier acc={float(summary.get('accuracy', 0.0)):.2%}, n={int(summary.get('samples', 0))}"

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(df))
    bars = ax.barh(y, df["accuracy"], color="#4C78A8", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["true_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.75, float(df["accuracy"].max()) + 0.08))
    ax.set_xlabel("Classification accuracy")
    title = "Per-class EEG Classifier Accuracy"
    if summary_text:
        title += f"\n{summary_text}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    for bar, count in zip(bars, df["count"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(count)}",
            va="center",
            fontsize=8,
            color="#444444",
        )
    savefig(fig, out_dir, "07_per_class_classification_accuracy")


def draw_qualitative_cases(out_dir: Path, csv_path: Path, n_cases: int = 4) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path).head(n_cases)
    if df.empty:
        return

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Object": row.get("true_object", ""),
                "Reference": row.get("reference_caption", ""),
                "High only": row.get("high_generated_caption", ""),
                "High + Reliable Low": row.get("reliable_generated_caption", ""),
                "F1 gain": f"{float(row.get('delta_token_f1', 0.0)):.3f}",
                "CIDEr gain": f"{float(row.get('delta_cider', 0.0)):.3f}",
            }
        )

    wrapped = []
    for row in rows:
        wrapped.append(
            [
                textwrap.fill(str(row["Object"]), 12),
                textwrap.fill(str(row["Reference"]), 38),
                textwrap.fill(str(row["High only"]), 38),
                textwrap.fill(str(row["High + Reliable Low"]), 38),
                row["F1 gain"],
                row["CIDEr gain"],
            ]
        )

    columns = ["Object", "Reference", "High only", "High + Reliable Low", "F1 gain", "CIDEr gain"]
    fig, ax = plt.subplots(figsize=(15, 2.2 + n_cases * 1.45))
    ax.axis("off")
    table = ax.table(
        cellText=wrapped,
        colLabels=columns,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.08, 0.25, 0.25, 0.25, 0.07, 0.07],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 3.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row == 0:
            cell.set_facecolor("#2F4B7C")
            cell.set_text_props(color="white", weight="bold", ha="center")
        elif col == 3:
            cell.set_facecolor("#EEF7EF")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F8FA")
        else:
            cell.set_facecolor("white")
    ax.set_title("Qualitative Examples: Reliable Low-level Evidence Improves Captions", pad=18, weight="bold")
    savefig(fig, out_dir, "08_qualitative_improved_cases")


def draw_semantic_diagnostics(out_dir: Path, hit_json: Path, agreement_json: Path) -> None:
    panels = []

    if hit_json.exists():
        hit = pd.read_json(hit_json, typ="series").to_dict()
        hit_items = []
        for level in ["low", "mid", "high"]:
            for key in [f"{level}_top1_hit", f"{level}_top3_hit", f"{level}_hit_rate"]:
                if key in hit:
                    hit_items.append((key.replace("_", " "), float(hit[key])))
                    break
        if hit_items:
            panels.append(("Semantic hit rate", hit_items))

    if agreement_json.exists():
        agr = pd.read_json(agreement_json, typ="series").to_dict()
        agr_items = []
        for key in ["low_eq_mid", "low_eq_high", "mid_eq_high", "all_equal"]:
            if key in agr:
                agr_items.append((key.replace("_", "=="), float(agr[key])))
        if agr_items:
            panels.append(("Head label agreement", agr_items))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, items) in zip(axes, panels):
        labels, values = zip(*items)
        x = np.arange(len(labels))
        ax.bar(x, values, color="#F28E2B")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Ratio")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for xi, value in zip(x, values):
            ax.text(xi, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    savefig(fig, out_dir, "09_semantic_retrieval_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/paper_figures_final")
    parser.add_argument("--source_vis_dir", default="outputs/paper_visualizations_low_high_reliable")
    parser.add_argument("--tsne_dir", default="outputs/paper_visualizations_tsne_full")
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_vis = Path(args.source_vis_dir)
    tsne_dir = Path(args.tsne_dir)

    draw_pipeline(out_dir)
    draw_metric_table(out_dir, source_vis / "main_result_table_rounded.csv")
    draw_per_class_accuracy(out_dir, tsne_dir / "tsne_test_per_class_accuracy.csv")
    draw_qualitative_cases(out_dir, source_vis / "top_improved_qualitative_cases.csv")
    draw_semantic_diagnostics(
        out_dir,
        Path("outputs/semantic_level_hit_random100_seed42_summary.json"),
        Path("outputs/semantic_level_label_agreement_full_summary.json"),
    )

    for name in [
        "main_metrics_grouped_bar.png",
        "main_metrics_grouped_bar.pdf",
        "main_ablation_metric_heatmap.png",
        "main_ablation_metric_heatmap.pdf",
        "relative_improvement_over_high_only.png",
        "relative_improvement_over_high_only.pdf",
    ]:
        copy_if_exists(source_vis / name, out_dir)

    for name in [
        "tsne_test_by_true_label.png",
        "tsne_test_by_true_label.pdf",
        "tsne_test_correct_vs_wrong.png",
        "tsne_test_correct_vs_wrong.pdf",
    ]:
        copy_if_exists(tsne_dir / name, out_dir)

    print(f"Saved all paper figures to {out_dir}")


if __name__ == "__main__":
    main()
