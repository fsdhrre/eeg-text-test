import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import PathConfig, TrainConfig
from eeg_text_codex.utils import get_device
from scripts.evaluate_caption_metrics import (
    cider_scores,
    corpus_bleu,
    meteor_like,
    rouge_l,
    sentence_bleu,
    strip_template_prefix,
    token_f1,
    tokenize,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate captions with text, visual, object, and evidence metrics.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--reference_col", default="reference_caption")
    parser.add_argument("--generated_col", default="generated_caption")
    parser.add_argument("--true_label_col", default="true_object")
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--image_name_col", default="image_name")
    parser.add_argument("--image_dir", default=PathConfig.image_dir)
    parser.add_argument("--clip_path", default=PathConfig.clip_path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--strip_template_prefix", action="store_true")
    parser.add_argument("--skip_image_clip", action="store_true")
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def label_variants(label):
    label = str(label).lower().strip()
    pieces = re.split(r"\s+or\s+|/", label)
    variants = set()
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        variants.add(piece)
        if piece.endswith("s"):
            variants.add(piece[:-1])
        else:
            variants.add(piece + "s")
    return variants


def object_accuracy(label, caption):
    caption_text = str(caption).lower()
    caption_tokens = set(tokenize(caption_text))
    for variant in label_variants(label):
        variant_tokens = tokenize(variant)
        if not variant_tokens:
            continue
        if len(variant_tokens) == 1 and variant_tokens[0] in caption_tokens:
            return 1.0
        if re.search(r"\b" + re.escape(variant) + r"\b", caption_text):
            return 1.0
    return 0.0


def content_tokens(text):
    tokens = tokenize(text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def extract_reliable_evidence(prompt):
    prompt = str(prompt)
    match = re.search(
        r"Reliable semantic cues retrieved from EEG:\s*(.*?)\n\s*Uncertain cues",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    evidence_lines = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        line = line.lstrip("-").strip()
        if line.lower() == "none":
            continue
        if " - " in line:
            line = line.split(" - ", 1)[1]
        evidence_lines.append(line)
    return evidence_lines


def evidence_token_set(prompt):
    evidence = extract_reliable_evidence(prompt)
    tokens = []
    for item in evidence:
        tokens.extend(content_tokens(item))
    return set(tokens)


def cue_coverage(prompt, caption):
    evidence_tokens = evidence_token_set(prompt)
    if not evidence_tokens:
        return float("nan")
    generated_tokens = set(content_tokens(caption))
    return len(evidence_tokens & generated_tokens) / len(evidence_tokens)


def evidence_faithfulness(prompt, caption, true_label="", main_label=""):
    generated_tokens = set(content_tokens(caption))
    if not generated_tokens:
        return float("nan")
    support = evidence_token_set(prompt)
    support.update(content_tokens(true_label))
    support.update(content_tokens(main_label))
    if not support:
        return float("nan")
    return len(generated_tokens & support) / len(generated_tokens)


def image_path_from_name(image_dir, image_name):
    image_name = str(image_name)
    return os.path.join(image_dir, image_name.split("_")[0], image_name + ".JPEG")


@torch.no_grad()
def encode_clip_texts(texts, tokenizer, model, device, batch_size):
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode", leave=False):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        chunks.append(F.normalize(model.get_text_features(**inputs).float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_clip_images(paths, processor, model, device, batch_size):
    chunks = []
    valid = []
    for start in tqdm(range(0, len(paths), batch_size), desc="CLIP image encode", leave=False):
        batch_paths = paths[start:start + batch_size]
        images = []
        batch_valid = []
        for path in batch_paths:
            if not os.path.exists(path):
                batch_valid.append(False)
                continue
            images.append(Image.open(path).convert("RGB"))
            batch_valid.append(True)
        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            embeds = F.normalize(model.get_image_features(**inputs).float(), dim=-1).cpu()
        else:
            embeds = torch.empty(0, model.config.projection_dim)
        cursor = 0
        for ok in batch_valid:
            if ok:
                chunks.append(embeds[cursor:cursor + 1])
                cursor += 1
            else:
                chunks.append(torch.full((1, model.config.projection_dim), float("nan")))
        valid.extend(batch_valid)
    return torch.cat(chunks, dim=0), valid


def mean_valid(values):
    clean = [float(v) for v in values if not pd.isna(v)]
    return float(sum(clean) / len(clean)) if clean else None


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

    df["token_f1_scored"] = [token_f1(r, h) for r, h in zip(references, hypotheses)]
    df["bleu1"] = [sentence_bleu(r, h, max_n=1) for r, h in zip(references, hypotheses)]
    df["bleu2"] = [sentence_bleu(r, h, max_n=2) for r, h in zip(references, hypotheses)]
    df["bleu3"] = [sentence_bleu(r, h, max_n=3) for r, h in zip(references, hypotheses)]
    df["bleu4"] = [sentence_bleu(r, h, max_n=4) for r, h in zip(references, hypotheses)]
    df["rouge_l"] = [rouge_l(r, h) for r, h in zip(references, hypotheses)]
    df["meteor"] = [meteor_like(r, h) for r, h in zip(references, hypotheses)]
    df["cider"] = cider_scores(references, hypotheses)

    true_labels = df[args.true_label_col].fillna("").astype(str).tolist()
    main_labels = df["main_label"].fillna("").astype(str).tolist() if "main_label" in df else [""] * len(df)
    prompts = df[args.prompt_col].fillna("").astype(str).tolist() if args.prompt_col in df else [""] * len(df)
    df["object_accuracy"] = [object_accuracy(label, hyp) for label, hyp in zip(true_labels, hypotheses)]
    df["cue_coverage"] = [cue_coverage(prompt, hyp) for prompt, hyp in zip(prompts, hypotheses)]
    df["evidence_faithfulness"] = [
        evidence_faithfulness(prompt, hyp, true_label, main_label)
        for prompt, hyp, true_label, main_label in zip(prompts, hypotheses, true_labels, main_labels)
    ]

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    clip_model = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(device).eval()
    ref_clip = encode_clip_texts(references, tokenizer, clip_model, device, args.batch_size)
    hyp_clip = encode_clip_texts(hypotheses, tokenizer, clip_model, device, args.batch_size)
    df["clip_text_similarity"] = (ref_clip * hyp_clip).sum(dim=-1).tolist()

    image_clip_available = False
    if not args.skip_image_clip:
        processor = CLIPImageProcessor.from_pretrained(args.clip_path, local_files_only=True)
        image_paths = [
            image_path_from_name(args.image_dir, image_name)
            for image_name in df[args.image_name_col].fillna("").astype(str).tolist()
        ]
        image_clip, image_valid = encode_clip_images(image_paths, processor, clip_model, device, args.batch_size)
        sims = (image_clip * hyp_clip).sum(dim=-1)
        df["image_text_clipscore"] = sims.tolist()
        df["image_found"] = image_valid
        image_clip_available = True
    else:
        df["image_text_clipscore"] = float("nan")
        df["image_found"] = False

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "input_csv": args.input_csv,
        "rows": int(len(df)),
        "strip_template_prefix": bool(args.strip_template_prefix),
        "image_clip_available": image_clip_available,
        "metrics": {
            "object_accuracy": float(df["object_accuracy"].mean()),
            "image_text_clipscore": mean_valid(df["image_text_clipscore"].tolist()),
            "clip_text_similarity": float(df["clip_text_similarity"].mean()),
            "token_f1": float(df["token_f1_scored"].mean()),
            "bleu1_sentence_avg": float(df["bleu1"].mean()),
            "bleu2_sentence_avg": float(df["bleu2"].mean()),
            "bleu3_sentence_avg": float(df["bleu3"].mean()),
            "bleu4_sentence_avg": float(df["bleu4"].mean()),
            "bleu4_corpus": corpus_bleu(references, hypotheses, max_n=4),
            "rouge_l": float(df["rouge_l"].mean()),
            "meteor": float(df["meteor"].mean()),
            "cider": float(df["cider"].mean()),
            "cue_coverage": mean_valid(df["cue_coverage"].tolist()),
            "evidence_faithfulness": mean_valid(df["evidence_faithfulness"].tolist()),
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-sample metrics to {args.output_csv}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
