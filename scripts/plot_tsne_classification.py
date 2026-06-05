import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig, TrainConfig
from eeg_text_codex.data import EEGImageDataset, collate_eeg_label_batch
from eeg_text_codex.utils import ensure_source_on_path, get_device, load_eeg_encoder


EEGPT_KWARGS = '{"img_size":[58,1024],"patch_size":64,"patch_stride":64,"embed_num":4,"embed_dim":512,"depth":8,"num_heads":8,"mlp_ratio":4.0,"qkv_bias":true}'


def parse_args():
    parser = argparse.ArgumentParser(description="Plot t-SNE for EEG classifier features.")
    parser.add_argument("--checkpoint_dir", default=os.path.join(PathConfig.staged_output_dir, "stage2_eegpt_structured", "best"))
    parser.add_argument("--output_dir", default=os.path.join(PathConfig.staged_output_dir, "paper_visualizations_tsne"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=-1, help="Use <=0 for all samples.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--eeg_encoder_type", choices=["channelnet", "eegpt"], default="eegpt")
    parser.add_argument("--eegpt_model_dir", default="external/EEGPT/downstream")
    parser.add_argument("--eegpt_checkpoint_path", default="external/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt")
    parser.add_argument("--eegpt_import", default="Modules.models.EEGPT_mcae:EEGTransformer")
    parser.add_argument("--eegpt_model_kwargs", default=EEGPT_KWARGS)
    parser.add_argument("--eegpt_backbone_out_dim", type=int, default=2048)
    parser.add_argument("--eeg_feature_dim", type=int, default=PathConfig.eeg_feature_dim)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def configure_paths(args):
    paths = PathConfig()
    paths.eeg_encoder_type = args.eeg_encoder_type
    paths.eeg_encoder_path = os.path.join(args.checkpoint_dir, "eeg_encoder")
    if not os.path.exists(os.path.join(paths.eeg_encoder_path, "encoder.pt")):
        paths.eeg_encoder_path = args.checkpoint_dir
    paths.eegpt_model_dir = args.eegpt_model_dir
    paths.eegpt_checkpoint_path = args.eegpt_checkpoint_path
    paths.eegpt_import = args.eegpt_import
    paths.eegpt_model_kwargs = args.eegpt_model_kwargs
    paths.eegpt_backbone_out_dim = args.eegpt_backbone_out_dim
    paths.eeg_feature_dim = args.eeg_feature_dim
    return paths


def collect_features(args, paths, device):
    data_cfg = DataConfig()
    dataset = EEGImageDataset(
        eeg_dataset=paths.eeg_dataset,
        splits_path=paths.splits_path,
        image_dir=paths.image_dir,
        split_name=args.split,
        split_num=data_cfg.split_num,
        time_low=data_cfg.time_low,
        time_high=data_cfg.time_high,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eeg_label_batch,
    )
    encoder = load_eeg_encoder(paths, device).eval()

    features = []
    labels = []
    preds = []
    image_names = []
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collect {args.split} EEG features"):
            eeg = batch["eeg"].unsqueeze(1).to(device)
            feat, logits = encoder(eeg)
            pred = logits.argmax(dim=-1).cpu()
            features.append(feat.float().cpu())
            labels.append(batch["labels"].cpu())
            preds.append(pred)
            image_names.extend(batch["image_names"])
            seen += eeg.size(0)
            if args.max_samples > 0 and seen >= args.max_samples:
                break

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    if args.max_samples > 0:
        features = features[:args.max_samples]
        labels = labels[:args.max_samples]
        preds = preds[:args.max_samples]
        image_names = image_names[:args.max_samples]
    return features.numpy(), labels.numpy(), preds.numpy(), image_names


def plot_tsne(coords, labels, preds, label_names, out_dir, split):
    correct = labels == preds
    acc = float(correct.mean())
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(11, 8.5))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.72,
            color=label_to_color[label],
            label=label_names.get(str(int(label)), str(label)),
            edgecolors="none",
        )
    wrong = ~correct
    if wrong.any():
        plt.scatter(
            coords[wrong, 0],
            coords[wrong, 1],
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            label="Misclassified",
        )
    plt.title(f"t-SNE of EEG Classifier Features ({split}, n={len(labels)}, classifier acc={acc:.2%})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(alpha=0.18)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=False,
        ncol=1,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"tsne_{split}_by_true_label.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"tsne_{split}_by_true_label.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6.5))
    plt.scatter(coords[correct, 0], coords[correct, 1], s=18, alpha=0.72, color="#4F9D69", label="Correct")
    plt.scatter(coords[wrong, 0], coords[wrong, 1], s=18, alpha=0.78, color="#B85C5C", label="Wrong")
    plt.title(f"t-SNE Correct vs Wrong ({split}, n={len(labels)}, classifier acc={acc:.2%})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(alpha=0.18)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"tsne_{split}_correct_vs_wrong.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"tsne_{split}_correct_vs_wrong.pdf", bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    paths = configure_paths(args)
    ensure_source_on_path(paths.source_root)
    from constants import id2label

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    features, labels, preds, image_names = collect_features(args, paths, device)
    perplexity = min(args.perplexity, max(5.0, (len(features) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )
    coords = tsne.fit_transform(features)

    rows = []
    for i, image_name in enumerate(image_names):
        rows.append(
            {
                "image_name": image_name,
                "true_label_id": int(labels[i]),
                "true_label": id2label[str(int(labels[i]))],
                "pred_label_id": int(preds[i]),
                "pred_label": id2label[str(int(preds[i]))],
                "correct": bool(labels[i] == preds[i]),
                "tsne_x": float(coords[i, 0]),
                "tsne_y": float(coords[i, 1]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"tsne_{args.split}_coordinates.csv", index=False)

    acc = float((labels == preds).mean())
    per_class = (
        df.groupby("true_label")
        .agg(count=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values("accuracy")
    )
    per_class.to_csv(out_dir / f"tsne_{args.split}_per_class_accuracy.csv", index=False)
    summary = {
        "split": args.split,
        "samples": int(len(df)),
        "checkpoint_dir": args.checkpoint_dir,
        "accuracy": acc,
        "perplexity": float(perplexity),
        "random_state": args.random_state,
    }
    with open(out_dir / f"tsne_{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_tsne(coords, labels, preds, id2label, out_dir, args.split)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved t-SNE plots and tables to {out_dir}")


if __name__ == "__main__":
    main()
