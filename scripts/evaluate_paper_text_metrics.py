import argparse
import json
import math
import os
import sys
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import TrainConfig
from eeg_text_codex.utils import get_device
from scripts.evaluate_caption_metrics import (
    meteor_like,
    rouge_l,
    sentence_bleu,
    strip_template_prefix,
    tokenize,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute paper text metrics: ROUGE-N/L, BLEU-N, METEOR, BERTScore.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--reference_col", default="reference_caption")
    parser.add_argument("--generated_col", default="generated_caption")
    parser.add_argument("--strip_template_prefix", action="store_true")
    parser.add_argument(
        "--bert_model_path",
        default="",
        help="Optional local BERT/RoBERTa/SBERT model path for BERTScore-like contextual token matching.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def rouge_n(reference, hypothesis, n):
    ref_ngrams = Counter(ngrams(tokenize(reference), n))
    hyp_ngrams = Counter(ngrams(tokenize(hypothesis), n))
    if not ref_ngrams or not hyp_ngrams:
        return 0.0, 0.0, 0.0
    overlap = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
    precision = overlap / sum(hyp_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mean(values):
    clean = [float(v) for v in values if not pd.isna(v)]
    return float(sum(clean) / len(clean)) if clean else None


@torch.no_grad()
def encode_for_bertscore(texts, tokenizer, model, device, batch_size):
    embeddings = []
    masks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="BERTScore encode", leave=False):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        hidden = F.normalize(outputs.last_hidden_state.float(), dim=-1).cpu()
        mask = inputs["attention_mask"].bool().cpu()
        embeddings.extend(hidden)
        masks.extend(mask)
    return embeddings, masks


def bertscore_pair(ref_emb, ref_mask, hyp_emb, hyp_mask):
    ref = ref_emb[ref_mask]
    hyp = hyp_emb[hyp_mask]
    if ref.numel() == 0 or hyp.numel() == 0:
        return 0.0, 0.0, 0.0
    sims = hyp @ ref.t()
    precision = float(sims.max(dim=1).values.mean().item())
    recall = float(sims.max(dim=0).values.mean().item())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def add_bertscore(df, references, hypotheses, model_path, device_name, batch_size):
    if not model_path:
        df["bertscore_precision"] = float("nan")
        df["bertscore_recall"] = float("nan")
        df["bertscore_f1"] = float("nan")
        return False

    device = get_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    ref_embs, ref_masks = encode_for_bertscore(references, tokenizer, model, device, batch_size)
    hyp_embs, hyp_masks = encode_for_bertscore(hypotheses, tokenizer, model, device, batch_size)
    scores = [
        bertscore_pair(ref_emb, ref_mask, hyp_emb, hyp_mask)
        for ref_emb, ref_mask, hyp_emb, hyp_mask in zip(ref_embs, ref_masks, hyp_embs, hyp_masks)
    ]
    df["bertscore_precision"] = [item[0] for item in scores]
    df["bertscore_recall"] = [item[1] for item in scores]
    df["bertscore_f1"] = [item[2] for item in scores]
    return True


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    references = df[args.reference_col].fillna("").astype(str).tolist()
    hypotheses = df[args.generated_col].fillna("").astype(str).tolist()
    if args.strip_template_prefix:
        references = [strip_template_prefix(text) for text in references]
        hypotheses = [strip_template_prefix(text) for text in hypotheses]
        df["reference_caption_scored"] = references
        df["generated_caption_scored"] = hypotheses

    for n in [1, 2]:
        scores = [rouge_n(ref, hyp, n) for ref, hyp in zip(references, hypotheses)]
        df[f"rouge{n}_precision"] = [item[0] for item in scores]
        df[f"rouge{n}_recall"] = [item[1] for item in scores]
        df[f"rouge{n}_f1"] = [item[2] for item in scores]

    df["rouge_l"] = [rouge_l(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    for n in [1, 2, 3, 4]:
        df[f"bleu{n}"] = [sentence_bleu(ref, hyp, max_n=n) for ref, hyp in zip(references, hypotheses)]
    df["meteor"] = [meteor_like(ref, hyp) for ref, hyp in zip(references, hypotheses)]

    bertscore_available = add_bertscore(
        df,
        references,
        hypotheses,
        args.bert_model_path,
        args.device,
        args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary = {
        "input_csv": args.input_csv,
        "rows": int(len(df)),
        "strip_template_prefix": bool(args.strip_template_prefix),
        "bertscore_available": bertscore_available,
        "bert_model_path": args.bert_model_path,
        "metrics": {
            "rouge1_f1": mean(df["rouge1_f1"].tolist()),
            "rouge2_f1": mean(df["rouge2_f1"].tolist()),
            "rouge_l": mean(df["rouge_l"].tolist()),
            "bleu1": mean(df["bleu1"].tolist()),
            "bleu2": mean(df["bleu2"].tolist()),
            "bleu3": mean(df["bleu3"].tolist()),
            "bleu4": mean(df["bleu4"].tolist()),
            "meteor": mean(df["meteor"].tolist()),
            "bertscore_precision": mean(df["bertscore_precision"].tolist()),
            "bertscore_recall": mean(df["bertscore_recall"].tolist()),
            "bertscore_f1": mean(df["bertscore_f1"].tolist()),
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-sample metrics to {args.output_csv}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
