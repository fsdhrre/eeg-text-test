from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


FEATURE_NAMES = [
    "low_best",
    "mid_best",
    "high_best",
    "low_mean3",
    "mid_mean3",
    "high_mean3",
    "low_top1",
    "mid_top1",
    "high_top1",
    "retrieval_top1_count",
    "classifier_prob",
    "classifier_rank_score",
    "classifier_top1",
    "classifier_in_topk",
]


def ensure_label_indices(db: Dict, device: torch.device) -> Dict:
    if "label_indices" in db:
        return db
    label_indices = {}
    for index, label in enumerate(db["label_names"]):
        label_indices.setdefault(label, []).append(index)
    db["label_indices"] = {
        label: torch.tensor(indices, dtype=torch.long, device=device)
        for label, indices in label_indices.items()
    }
    return db


def classifier_label_scores(cls_logits: torch.Tensor, id2label: Dict[str, str], top_k: int) -> Dict[str, Dict[str, float]]:
    probs = F.softmax(cls_logits.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()), dim=-1)
    scores = {}
    for rank, (value, index) in enumerate(zip(values, indices)):
        label = id2label[str(int(index.item()))]
        scores[label] = {
            "prob": float(value.item()),
            "rank_score": 1.0 / (rank + 1),
            "top1": 1.0 if rank == 0 else 0.0,
            "in_topk": 1.0,
        }
    return scores


def candidate_feature_tensor(
    predictions_by_level: Dict[str, torch.Tensor],
    db: Dict,
    candidate_labels: Iterable[str],
    cls_logits: torch.Tensor,
    id2label: Dict[str, str],
    classifier_top_k: int,
) -> Tuple[List[str], torch.Tensor]:
    device = next(iter(db["embeddings"].values())).device
    db = ensure_label_indices(db, device)
    labels = sorted(label for label in candidate_labels if label in db["label_indices"])
    if not labels:
        return [], torch.empty(0, len(FEATURE_NAMES), device=device)

    cls_scores = classifier_label_scores(cls_logits, id2label, classifier_top_k)
    level_best = {}
    level_mean3 = {}
    level_top1 = {}
    for level, pred in predictions_by_level.items():
        query = F.normalize(pred.float(), dim=-1)
        sims = (query @ db["embeddings"][level].t()).squeeze(0)
        top_index = int(sims.argmax().item())
        level_top1[level] = db["label_names"][top_index]
        level_best[level] = {}
        level_mean3[level] = {}
        for label in labels:
            indices = db["label_indices"][label]
            label_sims = sims[indices]
            k = min(3, label_sims.numel())
            top_values = torch.topk(label_sims, k=k, dim=0).values
            level_best[level][label] = float(top_values[0].item())
            level_mean3[level][label] = float(top_values.mean().item())

    rows = []
    for label in labels:
        cls = cls_scores.get(label, {})
        retrieval_top1_count = sum(1.0 for level in ("low", "mid", "high") if level_top1[level] == label)
        rows.append([
            level_best["low"][label],
            level_best["mid"][label],
            level_best["high"][label],
            level_mean3["low"][label],
            level_mean3["mid"][label],
            level_mean3["high"][label],
            1.0 if level_top1["low"] == label else 0.0,
            1.0 if level_top1["mid"] == label else 0.0,
            1.0 if level_top1["high"] == label else 0.0,
            retrieval_top1_count,
            cls.get("prob", 0.0),
            cls.get("rank_score", 0.0),
            cls.get("top1", 0.0),
            cls.get("in_topk", 0.0),
        ])

    return labels, torch.tensor(rows, dtype=torch.float32, device=device)
