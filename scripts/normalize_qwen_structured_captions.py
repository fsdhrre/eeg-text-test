"""Normalize an existing Qwen-VL caption cache.

Use this script after generation if the cache contains extra markdown,
multi-sentence answers, missing final punctuation, or inconsistent category
prefixes. It keeps the same image keys and rewrites only the caption text.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import PathConfig


def parse_args():
    """Define input/output cache paths."""

    parser = argparse.ArgumentParser(description="Clean existing Qwen structured caption cache.")
    parser.add_argument("--input_json", default=PathConfig.structured_caption_path)
    parser.add_argument("--output_json", default=os.path.join(PathConfig.staged_output_dir, "qwen_structured_captions_clean.json"))
    parser.add_argument("--output_csv", default=os.path.join(PathConfig.staged_output_dir, "qwen_structured_captions_clean.csv"))
    return parser.parse_args()


def clean_one_sentence(text: str) -> str:
    """Strip common LLM artifacts and keep the first sentence only."""

    cleaned = text.strip().replace("\r", "\n")
    for marker in ["###", "Tags:", "Answer:", "Caption:", "Q:", "\n"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    cleaned = re.sub(r"^\s*[-*\d.]+\s*", "", cleaned)
    cleaned = cleaned.strip("\"' ")
    end_positions = [cleaned.find(end) for end in [".", "!", "?"] if cleaned.find(end) != -1]
    if end_positions:
        cleaned = cleaned[: min(end_positions) + 1].strip()
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_caption(caption: str, category: str) -> str:
    """Ensure the caption explicitly contains the dataset category."""

    caption = clean_one_sentence(caption)
    if category.lower() not in caption.lower():
        fragment = caption.rstrip(".!?").strip()
        if fragment:
            fragment = fragment[0].lower() + fragment[1:]
            return f"This image shows {category}: {fragment}."
        return f"This image shows {category} in a visible scene."
    return caption


def main():
    args = parse_args()
    # JSON is the authoritative cache because it preserves image path/category
    # metadata; CSV is written only as a convenient table preview.
    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    cleaned = {}
    for image_name, record in records.items():
        category = record.get("category", "")
        caption = normalize_caption(record.get("caption", ""), category)
        cleaned[image_name] = {
            **record,
            "caption": caption,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["image_name", "category", "image_path", "caption"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_name, record in cleaned.items():
                writer.writerow({
                    "image_name": image_name,
                    "category": record.get("category", ""),
                    "image_path": record.get("image_path", ""),
                    "caption": record.get("caption", ""),
                })

    print(f"Saved cleaned JSON to {args.output_json}")
    if args.output_csv:
        print(f"Saved cleaned CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
