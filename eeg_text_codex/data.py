import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


def clean_caption(caption: str) -> str:
    """Remove dataset wrappers and keep the natural English caption."""

    return caption.replace("<s>", "").replace("</s>", "").strip()


def load_caption_map(caption_map_path: Optional[str]) -> Dict[str, str]:
    """Load optional generated captions keyed by image name."""

    if not caption_map_path:
        return {}
    with open(caption_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    captions = {}
    for image_name, value in raw.items():
        if isinstance(value, str):
            caption = value
        elif isinstance(value, dict):
            caption = value.get("caption") or value.get("structured_caption") or ""
        else:
            caption = ""
        if caption:
            captions[image_name] = clean_caption(caption)
    return captions


def build_prompt_text(instruction: str, object_label: Optional[str] = None) -> str:
    """Create a prompt with an `<image>` placeholder for the EEG embedding.

    If `object_label` is supplied, it must be a predicted label from EEG, not
    a ground-truth label. Keeping this rule avoids test-time information leak.
    """

    if object_label:
        return f"<image> {object_label} {instruction}"
    return f"<image> {instruction}"


def split_prompt_ids(tokenizer, prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the LLM chat template, then split around the EEG embedding slot."""

    if "<image>" not in prompt_text:
        prompt_text = f"<image> {prompt_text}"

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prefix, suffix = prompt_text.split("<image>", 1)
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).input_ids.long().squeeze(0)
    suffix_ids = tokenizer(
        suffix.strip(),
        add_special_tokens=False,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).input_ids.long().squeeze(0)
    return prefix_ids, suffix_ids


def pad_1d(sequences: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pad a list of 1D token ID tensors to a dense batch."""

    max_len = max(seq.numel() for seq in sequences)
    output = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if seq.numel() > 0:
            output[i, : seq.numel()] = seq
    return output


class EEGCaptionDataset(Dataset):
    """Return EEG samples plus their standard English image captions.

    This dataset does not load image pixels for the second stage. Captions are
    read from files next to the images, while the input signal is EEG-only.
    """

    def __init__(
        self,
        eeg_dataset: str,
        splits_path: str,
        image_dir: str,
        tokenizer,
        split_name: str,
        split_num: int,
        time_low: int,
        time_high: int,
        instruction: str,
        max_caption_tokens: int,
        caption_map_path: Optional[str] = None,
    ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.time_low = time_low
        self.time_high = time_high
        self.instruction = instruction
        self.max_caption_tokens = max_caption_tokens
        self.caption_map = load_caption_map(caption_map_path)

        loaded = torch.load(eeg_dataset, map_location="cpu")
        self.data = loaded["dataset"]
        self.images = loaded["images"]

        split_file = torch.load(splits_path, map_location="cpu")
        split_idx = split_file["splits"][split_num][split_name]
        self.indices = [
            i for i in split_idx
            if 450 <= self.data[i]["eeg"].size(1) <= 600
        ]
        print(f"Total examples in {split_name}: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict:
        idx = self.indices[item]
        sample = self.data[idx]

        eeg = sample["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]
        eeg = eeg.t().view(1, 128, self.time_high - self.time_low)

        image_name = self.images[sample["image"]]
        if image_name in self.caption_map:
            caption = self.caption_map[image_name]
        else:
            caption_path = os.path.join(
                self.image_dir,
                image_name.split("_")[0],
                image_name + "_caption.txt",
            )
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = clean_caption(f.readline())

        caption_ids = self.tokenizer(
            caption + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_caption_tokens,
            return_tensors="pt",
        ).input_ids.long().squeeze(0)

        return {
            "eeg": eeg,
            "caption_ids": caption_ids,
            "caption": caption,
            "image_name": image_name,
            "label": int(sample["label"]),
        }


def collate_caption_batch(batch: List[Dict], pad_token_id: int) -> Dict:
    """Collate EEG tensors and variable-length caption token IDs."""

    return {
        "eeg": torch.cat([item["eeg"] for item in batch], dim=0),
        "caption_ids": pad_1d([item["caption_ids"] for item in batch], pad_token_id),
        "captions": [item["caption"] for item in batch],
        "image_names": [item["image_name"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


class EEGImageDataset(Dataset):
    """Return paired EEG, class labels, and image pixels for staged training."""

    def __init__(
        self,
        eeg_dataset: str,
        splits_path: str,
        image_dir: str,
        split_name: str,
        split_num: int,
        time_low: int,
        time_high: int,
        processor=None,
        load_image: bool = False,
    ):
        self.image_dir = image_dir
        self.time_low = time_low
        self.time_high = time_high
        self.processor = processor
        self.load_image = load_image

        loaded = torch.load(eeg_dataset, map_location="cpu")
        self.data = loaded["dataset"]
        self.images = loaded["images"]

        split_file = torch.load(splits_path, map_location="cpu")
        split_idx = split_file["splits"][split_num][split_name]
        self.indices = [
            i for i in split_idx
            if 450 <= self.data[i]["eeg"].size(1) <= 600
        ]
        print(f"Total examples in {split_name}: {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict:
        idx = self.indices[item]
        sample = self.data[idx]

        eeg = sample["eeg"].float().t()
        eeg = eeg[self.time_low:self.time_high, :]
        eeg = eeg.t().view(1, 128, self.time_high - self.time_low)

        image_name = self.images[sample["image"]]
        image_path = os.path.join(
            self.image_dir,
            image_name.split("_")[0],
            image_name + ".JPEG",
        )

        output = {
            "eeg": eeg,
            "label": int(sample["label"]),
            "image_name": image_name,
            "image_path": image_path,
        }
        if self.load_image:
            image = Image.open(image_path).convert("RGB")
            if self.processor is None:
                output["image"] = image
            else:
                image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
                output["pixel_values"] = image_inputs["pixel_values"].squeeze(0)
        return output


def collate_eeg_label_batch(batch: List[Dict]) -> Dict:
    """Collate batches for EEG classification."""

    return {
        "eeg": torch.cat([item["eeg"] for item in batch], dim=0),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_names": [item["image_name"] for item in batch],
    }


def collate_eeg_image_batch(batch: List[Dict]) -> Dict:
    """Collate batches for EEG-image semantic alignment."""

    return {
        "eeg": torch.cat([item["eeg"] for item in batch], dim=0),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_names": [item["image_name"] for item in batch],
    }
