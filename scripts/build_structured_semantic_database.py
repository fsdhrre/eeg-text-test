"""构建结构化 low / mid / high 语义目标数据库。

输入：
    Qwen-VL 生成的 JSON caption 缓存。每张图片对应一句规范化后的结构化描述，
    其中包含物体类别、视觉属性、动作/状态以及场景上下文。

输出：
    一个 ``.pt`` 数据库，里面保存 low、mid、high、full 四种文本视角的 CLIP
    text embedding。阶段二把这些 embedding 当作对比学习监督目标，阶段四把同一个
    数据库当作语义检索索引。
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig, TrainConfig
from eeg_text_codex.data import clean_caption, load_caption_map
from eeg_text_codex.utils import ensure_source_on_path, get_device


COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "gray", "grey",
    "silver", "gold", "golden", "dark", "bright", "colorful", "striped", "spotted", "transparent", "clear",
}
ATTRIBUTE_WORDS = {
    "large", "small", "old", "young", "modern", "vintage", "wooden", "metal", "leather", "plastic", "polished",
    "shiny", "glossy", "round", "square", "long", "short", "flat", "curved", "open", "closed", "smooth",
}
SCENE_WORDS = {
    "room", "restaurant", "street", "road", "field", "grass", "forest", "river", "water", "beach", "sky",
    "background", "table", "floor", "desk", "wall", "zoo", "indoor", "outdoor", "city", "garage", "studio",
    "track", "mountain", "hill", "snow", "lake", "park", "shelf", "kitchen", "bar",
}
ACTION_PATTERNS = [
    r"\b(?:is|are|was|were)\s+([a-z]+ing)\b",
    r"\b(?:sits|stands|rides|plays|holds|wears|flies|swims|moves|rests|lies|jumps|runs|walks|poses|navigates)\b",
]


def parse_args():
    """收集构建语义数据库所需的路径和 CLIP 配置。"""

    parser = argparse.ArgumentParser(description="Build explicit low/mid/high text semantic targets.")
    parser.add_argument("--split_names", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--caption_map_path", default=os.path.join(PathConfig.staged_output_dir, "qwen_structured_captions_v3.json"))
    parser.add_argument("--output_path", default=os.path.join(PathConfig.staged_output_dir, "structured_semantic_targets_all.pt"))
    parser.add_argument("--clip_path", default=PathConfig.clip_path)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def strip_caption_prefix(caption):
    """去掉规范化 Qwen caption 中固定的类别前缀。

    生成器保存的文本类似 ``The category is piano; it shows ...``。
    做语义分解时，类别单独保留，后面的描述片段用于抽取属性、动作和场景词。
    """

    match = re.match(r"^The category is (.*?);\s*it shows\s*(.*)$", caption.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip().rstrip(".")
    return "", caption.strip().rstrip(".")


def select_words(text, vocabulary):
    """从描述中筛出预定义词表里的属性词或场景词。"""

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return sorted({token for token in tokens if token in vocabulary})


def extract_actions(text):
    """用轻量正则抽取简单动作/状态线索。"""

    actions = []
    lower = text.lower()
    for pattern in ACTION_PATTERNS:
        for match in re.finditer(pattern, lower):
            if match.groups():
                actions.append(match.group(1))
            else:
                actions.append(match.group(0))
    return sorted(set(actions))


def structured_texts(label_name, caption):
    """把一句 caption 转成 low / mid / high 三种语义文本。

    low：颜色、材质、大小、纹理等视觉属性。
    mid：主体物体以及动作/状态。
    high：场景、背景和上下文。

    如果某个层级没有抽到显式词，就回退使用完整描述，保证每张图都有可用的
    CLIP 文本嵌入。
    """

    category, description = strip_caption_prefix(caption)
    label = label_name or category
    colors = select_words(description, COLOR_WORDS)
    attributes = select_words(description, ATTRIBUTE_WORDS)
    scenes = select_words(description, SCENE_WORDS)
    actions = extract_actions(description)

    low_terms = ", ".join(colors + attributes) if colors or attributes else description
    mid_terms = ", ".join(actions) if actions else description
    high_terms = ", ".join(scenes) if scenes else description

    return {
        "low_text": f"Visual attributes of the {label}: {low_terms}.",
        "mid_text": f"Main object and action: {label}; {mid_terms}.",
        "high_text": f"Scene and context for the {label}: {high_terms}.",
        "full_text": caption,
        "low_terms": colors + attributes,
        "mid_terms": actions,
        "high_terms": scenes,
    }


def collect_rows(paths, data_cfg, split_names, caption_map):
    """收集 EEG split 中实际用到的唯一图片，并为每张图附上三层语义文本。"""

    loaded = torch.load(paths.eeg_dataset, map_location="cpu")
    data = loaded["dataset"]
    images = loaded["images"]
    split_file = torch.load(paths.splits_path, map_location="cpu")
    rows = {}

    ensure_source_on_path(paths.source_root)
    from constants import id2label

    for split_name in split_names:
        split_idx = split_file["splits"][data_cfg.split_num][split_name]
        for idx in split_idx:
            sample = data[idx]
            # 和训练预处理规则保持一致：只保留 EEG 长度落在有效视觉刺激窗口内的 trial。
            if not (450 <= sample["eeg"].size(1) <= 600):
                continue
            image_name = images[sample["image"]]
            if image_name in rows:
                rows[image_name]["splits"].add(split_name)
                continue
            caption = caption_map.get(image_name)
            if not caption:
                # 如果 Qwen 缓存里缺失该图片，就回退读取原始 caption，保证数据库完整。
                caption_path = os.path.join(paths.image_dir, image_name.split("_")[0], image_name + "_caption.txt")
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = clean_caption(f.readline())
            label = int(sample["label"])
            label_name = id2label[str(label)]
            texts = structured_texts(label_name, caption)
            rows[image_name] = {
                "image_name": image_name,
                "label": label,
                "label_name": label_name,
                "caption": caption,
                "splits": {split_name},
                **texts,
            }

    ordered = list(rows.values())
    for row in ordered:
        row["splits"] = sorted(row["splits"])
    return ordered


@torch.no_grad()
def encode_texts(rows, tokenizer, text_model, batch_size, device):
    """使用 CLIP text encoder 编码 low / mid / high / full 四种文本。"""

    embeddings = {}
    for field, key in [("low_text", "low"), ("mid_text", "mid"), ("high_text", "high"), ("full_text", "full")]:
        chunks = []
        texts = [row[field] for row in rows]
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Encode {key} semantics"):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            embeds = text_model(**inputs).text_embeds
            chunks.append(F.normalize(embeds.float(), dim=-1).cpu())
        embeddings[key] = torch.cat(chunks, dim=0)
    return embeddings


def main():
    args = parse_args()
    paths = PathConfig()
    data_cfg = DataConfig()
    caption_map = load_caption_map(args.caption_map_path)
    rows = collect_rows(paths, data_cfg, args.split_names, caption_map)
    print(f"Structured semantic rows: {len(rows)}")

    # CLIP embedding 定义了共享语义空间：阶段二用于对齐，阶段四用于检索。
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    text_model = CLIPTextModelWithProjection.from_pretrained(args.clip_path, local_files_only=True).to(device).eval()
    embeddings = encode_texts(rows, tokenizer, text_model, args.batch_size, device)

    # 同时保存可读文本和 tensor embedding：可读文本用于构造 evidence prompt，
    # tensor embedding 用于快速相似度检索。
    payload = {
        "metadata": {
            "split_names": args.split_names,
            "caption_map_path": args.caption_map_path,
            "clip_path": args.clip_path,
            "semantic_type": "structured_text",
        },
        "image_names": [row["image_name"] for row in rows],
        "captions": [row["caption"] for row in rows],
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "label_names": [row["label_name"] for row in rows],
        "splits": [row["splits"] for row in rows],
        "texts": {
            "low": [row["low_text"] for row in rows],
            "mid": [row["mid_text"] for row in rows],
            "high": [row["high_text"] for row in rows],
            "full": [row["full_text"] for row in rows],
        },
        "terms": {
            "low": [row["low_terms"] for row in rows],
            "mid": [row["mid_terms"] for row in rows],
            "high": [row["high_terms"] for row in rows],
        },
        "embeddings": embeddings,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(payload, args.output_path)
    sidecar_path = os.path.splitext(args.output_path)[0] + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": payload["metadata"],
            "num_images": len(rows),
            "first_rows": [
                {
                    "image_name": row["image_name"],
                    "label": row["label_name"],
                    "low": row["low_text"],
                    "mid": row["mid_text"],
                    "high": row["high_text"],
                    "caption": row["caption"],
                }
                for row in rows[:10]
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved structured semantic targets to {args.output_path}")
    print(f"Saved summary to {sidecar_path}")


if __name__ == "__main__":
    main()
