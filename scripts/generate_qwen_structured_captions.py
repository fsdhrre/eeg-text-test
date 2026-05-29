"""Generate structured one-sentence image captions with Qwen-VL.

This is an offline data-preparation script. It visits every unique image used
by the EEG train/val/test splits, sends the image directly into Qwen-VL-Chat,
and caches a concise English description. The cached captions become the source
text for the low/mid/high semantic database.

The script is resumable: existing records are loaded and skipped unless
``--overwrite`` is set.
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig
from eeg_text_codex.utils import ensure_source_on_path


PROMPT_TEMPLATE = (
    "You are creating a training caption for an EEG-to-text model. "
    "Describe only the visible action or state, visual attributes, and scene of the image. "
    "Do not name the object category and do not guess a label. "
    "Write exactly one natural English sentence of 10 to 22 words. "
    "Mention at least two visual attributes such as color, shape, material, count, or texture. "
    "Mention the scene or background. "
    "Do not copy this instruction. Do not use brackets, quotes, markdown, bullets, explanations, "
    "or multiple sentences."
)


def parse_args():
    """Define local model/data/cache options."""

    parser = argparse.ArgumentParser(
        description="Generate one-sentence structured image captions with local Qwen-VL-Chat."
    )
    parser.add_argument("--qwen_vl_path", default=PathConfig.qwen_vl_path)
    parser.add_argument("--eeg_dataset", default=PathConfig.eeg_dataset)
    parser.add_argument("--splits_path", default=PathConfig.splits_path)
    parser.add_argument("--image_dir", default=PathConfig.image_dir)
    parser.add_argument("--output_json", default=PathConfig.structured_caption_path)
    parser.add_argument(
        "--output_csv",
        default=os.path.join(PathConfig.staged_output_dir, "qwen_structured_captions.csv"),
    )
    parser.add_argument("--split_num", type=int, default=DataConfig.split_num)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--time_low", type=int, default=DataConfig.time_low)
    parser.add_argument("--time_high", type=int, default=DataConfig.time_high)
    parser.add_argument("--limit", type=int, default=-1, help="Generate only the first N missing images.")
    parser.add_argument("--dry_run", action="store_true", help="Only print dataset counts; do not load Qwen.")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_one_sentence(text: str) -> str:
    """Keep a single plain English sentence from Qwen's answer."""

    cleaned = text.strip().replace("\r", "\n")
    for marker in ["###", "Tags:", "Answer:", "Caption:", "Q:", "\n"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    cleaned = re.sub(r"^\s*[-*\d.]+\s*", "", cleaned)
    cleaned = cleaned.strip("\"' ")
    cleaned = cleaned.replace("[", "").replace("]", "")

    end_positions = [cleaned.find(end) for end in [".", "!", "?"] if cleaned.find(end) != -1]
    if end_positions:
        cleaned = cleaned[: min(end_positions) + 1].strip()
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned = cleaned + "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_caption_for_category(caption: str, category: str) -> str:
    """Make generated captions consistently start with the dataset category."""

    # Qwen is asked not to repeat the category. We add a deterministic prefix
    # afterward so every caption has the same schema and the label cannot drift.
    caption = clean_one_sentence(caption)
    expected_prefix = f"The category is {category};"
    fragment = caption.rstrip(".!?").strip()
    bad_fragments = [
        "main action or state",
        "two or more visual attributes",
        "scene or background",
        "visible action or state",
    ]
    if any(bad in fragment.lower() for bad in bad_fragments):
        fragment = ""
    if fragment:
        fragment = re.sub(r"^(the category is\s+[^;]+;\s*)", "", fragment, flags=re.IGNORECASE).strip()
        fragment = re.sub(r"^(it shows\s+)", "", fragment, flags=re.IGNORECASE).strip()
        fragment = fragment[0].lower() + fragment[1:]
        caption = f"{expected_prefix} it shows {fragment}."
    else:
        caption = f"{expected_prefix} it shows the object clearly with visible attributes in the scene."
    return caption


def load_existing(path: str) -> Dict[str, Dict]:
    """Load a partially generated cache if it exists."""

    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def save_outputs(records: Dict[str, Dict], json_path: str, csv_path: str) -> None:
    """Write both JSON cache and CSV preview for manual inspection."""

    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    if csv_path:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["image_name", "category", "image_path", "caption"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for image_name, record in records.items():
                writer.writerow({
                    "image_name": image_name,
                    "category": record.get("category", ""),
                    "image_path": record.get("image_path", ""),
                    "caption": record.get("caption", ""),
                })


def iter_used_images(args, label_map: Dict[str, str]) -> List[Dict[str, str]]:
    """Return the unique image files referenced by selected EEG splits."""

    loaded = torch.load(args.eeg_dataset, map_location="cpu")
    dataset = loaded["dataset"]
    images = loaded["images"]

    split_file = torch.load(args.splits_path, map_location="cpu")
    seen = set()
    items = []
    selected_splits = ["train", "val", "test"] if "all" in args.splits else args.splits
    for split_name in selected_splits:
        split_idx = split_file["splits"][args.split_num][split_name]
        for idx in split_idx:
            # Keep the same EEG-length filter as the training datasets so the
            # caption cache exactly matches usable EEG samples.
            if not 450 <= dataset[idx]["eeg"].size(1) <= 600:
                continue
            image_name = images[dataset[idx]["image"]]
            if image_name in seen:
                continue
            seen.add(image_name)
            synset = image_name.split("_")[0]
            items.append({
                "image_name": image_name,
                "category": label_map.get(synset, synset),
                "image_path": os.path.join(args.image_dir, synset, image_name + ".JPEG"),
            })
    return items


def load_qwen(args):
    """Load local Qwen-VL-Chat from disk.

    The cwd switch is kept for Qwen-VL implementations that expect relative
    imports/resources inside the model directory.
    """

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from modelscope import AutoModelForCausalLM, AutoTokenizer

    cwd = os.getcwd()
    os.chdir(args.qwen_vl_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.qwen_vl_path, trust_remote_code=True)
        load_kwargs = {
            "device_map": args.device_map,
            "trust_remote_code": True,
        }
        if args.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(args.qwen_vl_path, **load_kwargs).eval()
    finally:
        os.chdir(cwd)
    return tokenizer, model


def generate_caption(tokenizer, model, image_path: str, category: str) -> str:
    """Send the image path directly to Qwen-VL-Chat with one instruction."""

    prompt = PROMPT_TEMPLATE.format(category=category)
    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return clean_one_sentence(response)


def main():
    args = parse_args()
    ensure_source_on_path(PathConfig.source_root)
    from constants import label_map

    # Build the worklist first. This is cheap and supports --dry_run to verify
    # paths before loading the large VLM.
    items = iter_used_images(args, label_map)
    records = {} if args.overwrite else load_existing(args.output_json)
    missing = [item for item in items if args.overwrite or item["image_name"] not in records]
    if args.limit >= 0:
        missing = missing[: args.limit]

    print(f"Unique images from splits: {len(items)}")
    print(f"Already generated: {len(records) if not args.overwrite else 0}")
    print(f"Images to generate now: {len(missing)}")
    if args.dry_run:
        return
    if not missing:
        save_outputs(records, args.output_json, args.output_csv)
        print(f"Nothing new to generate. Saved current cache to {args.output_json}")
        return

    tokenizer, model = load_qwen(args)

    for step, item in enumerate(tqdm(missing, desc="Qwen structured captions"), start=1):
        if not os.path.exists(item["image_path"]):
            raise FileNotFoundError(item["image_path"])
        # The model sees the image and produces a description; the category is
        # enforced afterward from the dataset label to avoid random label names.
        caption = generate_caption(tokenizer, model, item["image_path"], item["category"])
        caption = normalize_caption_for_category(caption, item["category"])
        records[item["image_name"]] = {
            "caption": caption,
            "category": item["category"],
            "image_path": item["image_path"],
        }
        if args.save_every > 0 and step % args.save_every == 0:
            save_outputs(records, args.output_json, args.output_csv)

    save_outputs(records, args.output_json, args.output_csv)
    print(f"Saved {len(records)} structured captions to {args.output_json}")
    if args.output_csv:
        print(f"Saved readable CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
