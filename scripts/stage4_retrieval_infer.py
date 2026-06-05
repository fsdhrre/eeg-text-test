import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig, TrainConfig
from eeg_text_codex.data import EEGCaptionDataset, collate_caption_batch
from eeg_text_codex.modules import CandidateReranker, MultiHead
from eeg_text_codex.rerank import FEATURE_NAMES, candidate_feature_tensor
from eeg_text_codex.utils import ensure_source_on_path, get_device, load_eeg_encoder, load_llm_model


EEGPT_KWARGS = '{"img_size":[58,1024],"patch_size":64,"patch_stride":64,"embed_num":4,"embed_dim":512,"depth":8,"num_heads":8,"mlp_ratio":4.0,"qkv_bias":true}'


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 retrieval: retrieve semantic anchors and let LLM compose a caption.")
    parser.add_argument("--checkpoint_dir", default=os.path.join(PathConfig.staged_output_dir, "stage2_eegpt_retrieval", "best"))
    parser.add_argument("--semantic_db_path", default=os.path.join(PathConfig.staged_output_dir, "semantic_db_train.pt"))
    parser.add_argument("--output_csv", default=os.path.join(PathConfig.staged_output_dir, "stage4_retrieval_generations.csv"))
    parser.add_argument("--caption_map_path", default=os.path.join(PathConfig.staged_output_dir, "qwen_structured_captions_v3.json"))
    parser.add_argument("--llm_path", default=PathConfig.llm_path, help="Local LLM path used for caption generation.")
    parser.add_argument("--llm_trust_remote_code", action="store_true", help="Pass trust_remote_code=True when loading tokenizer/model.")
    parser.add_argument("--eeg_encoder_type", choices=["channelnet", "eegpt"], default="eegpt")
    parser.add_argument("--eegpt_model_dir", default="external/EEGPT/downstream")
    parser.add_argument("--eegpt_checkpoint_path", default="external/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt")
    parser.add_argument("--eegpt_import", default="Modules.models.EEGPT_mcae:EEGTransformer")
    parser.add_argument("--eegpt_model_kwargs", default=EEGPT_KWARGS)
    parser.add_argument("--eegpt_backbone_out_dim", type=int, default=2048)
    parser.add_argument("--eeg_feature_dim", type=int, default=PathConfig.eeg_feature_dim)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--classifier_top_k", type=int, default=5)
    parser.add_argument("--label_filter", choices=["none", "candidate"], default="none")
    parser.add_argument(
        "--semantic_levels",
        nargs="+",
        choices=["low", "mid", "high"],
        default=["low", "mid", "high"],
        help="Semantic levels used for retrieval/evidence. Use this for ablations, e.g. --semantic_levels mid high.",
    )
    parser.add_argument(
        "--decoupled_semantics",
        action="store_true",
        help=(
            "Use this when low/high targets no longer contain object labels. "
            "The main category then comes from the classifier, while low/high are exposed only as attribute/context cues."
        ),
    )
    parser.add_argument(
        "--classifier_only",
        action="store_true",
        help=(
            "Ablation mode: do not retrieve low/mid/high semantic evidence. "
            "The LLM receives only the classifier-based main category decision."
        ),
    )
    parser.add_argument("--reranker_path", default="", help="Optional Stage 5 reranker checkpoint directory or reranker.pt path.")
    parser.add_argument(
        "--anchor_mode",
        choices=["top1_per_level", "topk_per_level", "global_topk", "vote", "decision", "evidence"],
        default="top1_per_level",
        help="Control how many retrieved anchors are exposed to the LLM prompt.",
    )
    parser.add_argument(
        "--main_label_source",
        choices=["evidence", "classifier"],
        default="evidence",
        help=(
            "How to choose the object category used by the prompt. "
            "'evidence' lets retrieval evidence vote for the main label; "
            "'classifier' fixes the main label to the EEG classifier prediction and uses retrieval only for attributes/context."
        ),
    )
    parser.add_argument("--prompt_top_k", type=int, default=3, help="Used only by --anchor_mode global_topk.")
    parser.add_argument("--reliable_threshold", type=float, default=0.30)
    parser.add_argument(
        "--low_reliable_threshold",
        type=float,
        default=None,
        help="Optional level-specific threshold for low semantic evidence. Falls back to --reliable_threshold.",
    )
    parser.add_argument(
        "--mid_reliable_threshold",
        type=float,
        default=None,
        help="Optional level-specific threshold for mid semantic evidence. Falls back to --reliable_threshold.",
    )
    parser.add_argument(
        "--high_reliable_threshold",
        type=float,
        default=None,
        help="Optional level-specific threshold for high semantic evidence. Falls back to --reliable_threshold.",
    )
    parser.add_argument(
        "--drop_unreliable_low",
        action="store_true",
        help="For reliability-aware ablation: remove low-level cues below the low threshold instead of listing them as uncertain.",
    )
    parser.add_argument(
        "--evidence_margin_threshold",
        type=float,
        default=0.10,
        help="For --anchor_mode evidence, keep the old decision when candidate evidence is too close.",
    )
    parser.add_argument("--max_samples", type=int, default=20, help="Use <=0 for full test set.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument(
        "--generation_prompt_style",
        choices=["conservative", "structured"],
        default="conservative",
        help="structured matches the reference-caption template to improve lexical metrics.",
    )
    parser.add_argument("--skip_llm", action="store_true", help="Only write retrieved anchors, without generation.")
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def token_f1(reference: str, generated: str) -> float:
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    if not ref_tokens or not gen_tokens:
        return 0.0
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in gen_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def clean_generated_caption(text: str) -> str:
    cleaned = text.strip()
    for marker in ["###", "Tags:", "Answer:", "Q:", "\n"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    end_positions = [cleaned.find(end) for end in [".", "!", "?"] if cleaned.find(end) != -1]
    if end_positions:
        cleaned = cleaned[: min(end_positions) + 1].strip()
    return cleaned


def configure_paths(args):
    paths = PathConfig()
    paths.eeg_encoder_type = args.eeg_encoder_type
    paths.eeg_encoder_path = os.path.join(args.checkpoint_dir, "eeg_encoder")
    paths.eegpt_model_dir = args.eegpt_model_dir
    paths.eegpt_checkpoint_path = args.eegpt_checkpoint_path
    paths.eegpt_import = args.eegpt_import
    paths.eegpt_model_kwargs = args.eegpt_model_kwargs
    paths.eegpt_backbone_out_dim = args.eegpt_backbone_out_dim
    paths.eeg_feature_dim = args.eeg_feature_dim
    paths.llm_path = args.llm_path
    return paths


def load_retrieval_model(args, device):
    paths = configure_paths(args)
    eeg_encoder = load_eeg_encoder(paths, device).eval()
    multi_head = MultiHead(args.eeg_feature_dim, 512).to(device).eval()
    multi_head.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "multi_head.pt"), map_location=device))
    return eeg_encoder, multi_head


def load_reranker(path, device):
    if not path:
        return None
    checkpoint_path = path
    if os.path.isdir(path):
        checkpoint_path = os.path.join(path, "reranker.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_names = checkpoint.get("feature_names", FEATURE_NAMES)
    metadata = checkpoint.get("metadata", {})
    hidden_dim = int(metadata.get("hidden_dim", 64))
    dropout = float(metadata.get("dropout", 0.1))
    reranker = CandidateReranker(len(feature_names), hidden_dim=hidden_dim, dropout=dropout).to(device).eval()
    reranker.load_state_dict(checkpoint["state_dict"])
    return reranker


def load_semantic_db(path, device):
    db = torch.load(path, map_location="cpu")
    db["embeddings"] = {
        key: F.normalize(value.float(), dim=-1).to(device)
        for key, value in db["embeddings"].items()
    }
    db["labels"] = db["labels"].long()
    label_indices = {}
    for index, label in enumerate(db["label_names"]):
        label_indices.setdefault(label, []).append(index)
    db["label_indices"] = {
        label: torch.tensor(indices, dtype=torch.long, device=device)
        for label, indices in label_indices.items()
    }
    return db


def retrieve_one(query, db_embeds, top_k):
    query = F.normalize(query.float(), dim=-1)
    scores = query @ db_embeds.t()
    values, indices = torch.topk(scores, k=min(top_k, db_embeds.size(0)), dim=-1)
    return values.detach().cpu(), indices.detach().cpu()


def retrieve_one_filtered(query, db_embeds, candidate_mask, top_k):
    query = F.normalize(query.float(), dim=-1)
    scores = query @ db_embeds.t()
    scores = scores.masked_fill(~candidate_mask.unsqueeze(0), -1e9)
    valid_count = int(candidate_mask.sum().item())
    if valid_count <= 0:
        return retrieve_one(query, db_embeds, top_k)
    values, indices = torch.topk(scores, k=min(top_k, valid_count), dim=-1)
    return values.detach().cpu(), indices.detach().cpu()


def format_anchor(db, index, score, level):
    label_name = db["label_names"][index]
    caption = db["captions"][index]
    semantic_text = caption
    if "texts" in db and level in db["texts"]:
        semantic_text = db["texts"][level][index]
    image_name = db["image_names"][index]
    return {
        "label": label_name,
        "caption": caption,
        "semantic_text": semantic_text,
        "image_name": image_name,
        "score": float(score),
    }


def classifier_top_labels(cls_logits, id2label, top_k):
    values, indices = torch.topk(cls_logits.float(), k=min(top_k, cls_logits.numel()), dim=-1)
    return {id2label[str(int(index.item()))] for index in indices}


def classifier_top_label_scores(cls_logits, id2label, top_k):
    probs = F.softmax(cls_logits.float(), dim=-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()), dim=-1)
    scores = {}
    for rank, (value, index) in enumerate(zip(values, indices)):
        label = id2label[str(int(index.item()))]
        # Keep a rank bonus because EEG classifiers are often poorly calibrated.
        scores[label] = float(value.item()) + 0.20 / (rank + 1)
    return scores


def top_retrieval_labels(predictions_by_level, db, top_k):
    labels = set()
    for level, pred in predictions_by_level.items():
        scores, indices = retrieve_one(pred, db["embeddings"][level], top_k)
        for index in indices[0]:
            labels.add(db["label_names"][int(index)])
    return labels


def make_candidate_mask(db, candidate_labels, device):
    mask = torch.tensor([label in candidate_labels for label in db["label_names"]], dtype=torch.bool, device=device)
    return mask


def select_vote_anchors(anchors_by_level, prompt_top_k):
    level_weights = {"low": 0.7, "mid": 1.2, "high": 1.0}
    label_scores = {}
    label_best = {}
    for level, anchors in anchors_by_level.items():
        for rank, anchor in enumerate(anchors):
            rank_weight = 1.0 / (rank + 1)
            vote_score = level_weights.get(level, 1.0) * rank_weight * anchor["score"]
            label = anchor["label"]
            label_scores[label] = label_scores.get(label, 0.0) + vote_score
            previous = label_best.get(label)
            if previous is None or anchor["score"] > previous[1]["score"]:
                label_best[label] = (level, anchor)

    ranked_labels = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    selected = {"low": [], "mid": [], "high": []}
    used_images = set()
    for label, _ in ranked_labels[:prompt_top_k]:
        level, anchor = label_best[label]
        selected[level].append(anchor)
        used_images.add(anchor["image_name"])

    # Add mid/high top1 when not already represented; they often carry the
    # most useful object and scene cues for captioning.
    for level in ["mid", "high", "low"]:
        anchors = anchors_by_level.get(level, [])
        if anchors and anchors[0]["image_name"] not in used_images:
            selected[level].append(anchors[0])
            used_images.add(anchors[0]["image_name"])
    return selected


def decide_main_label(anchors_by_level, classifier_pred):
    top = {level: anchors[0] for level, anchors in anchors_by_level.items() if anchors}
    if not top:
        return classifier_pred, "fallback_classifier"
    low_label = top.get("low", {}).get("label")
    mid_label = top.get("mid", {}).get("label")
    high_label = top.get("high", {}).get("label")

    if mid_label and high_label and mid_label == high_label:
        return mid_label, "mid_high_agree"
    if classifier_pred in {label for label in [low_label, mid_label, high_label] if label}:
        return classifier_pred, "classifier_agrees_with_retrieval"

    label_counts = {}
    weighted_scores = {}
    level_weights = {"low": 0.7, "mid": 1.2, "high": 1.0}
    for level, anchors in anchors_by_level.items():
        for rank, anchor in enumerate(anchors):
            label = anchor["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            weighted_scores[label] = weighted_scores.get(label, 0.0) + level_weights.get(level, 1.0) * anchor["score"] / (rank + 1)

    repeated = [label for label, count in label_counts.items() if count >= 2]
    if repeated:
        repeated.sort(key=lambda label: weighted_scores[label], reverse=True)
        return repeated[0], "topk_repeated_label"

    if mid_label:
        return mid_label, "fallback_mid"
    best_level, best_anchor = max(top.items(), key=lambda item: item[1]["score"])
    return best_anchor["label"], f"fallback_{best_level}"


def score_candidate_labels(predictions_by_level, db, candidate_labels, cls_logits, id2label, classifier_top_k):
    labels = sorted(label for label in candidate_labels if label in db["label_indices"])
    if not labels:
        return {}

    level_weights = {"low": 0.75, "mid": 1.10, "high": 1.00}
    scores = {label: 0.0 for label in labels}
    parts = {label: {} for label in labels}
    for level, pred in predictions_by_level.items():
        query = F.normalize(pred.float(), dim=-1)
        sims = query @ db["embeddings"][level].t()
        for label in labels:
            indices = db["label_indices"][label]
            best_score = float(sims[:, indices].max().item())
            scores[label] += level_weights.get(level, 1.0) * best_score
            parts[label][level] = best_score

    cls_scores = classifier_top_label_scores(cls_logits, id2label, classifier_top_k)
    for label, cls_score in cls_scores.items():
        if label in scores:
            scores[label] += 0.65 * cls_score
            parts[label]["cls"] = cls_score

    return {
        label: {
            "score": scores[label],
            "parts": parts[label],
        }
        for label in labels
    }


def classifier_only_evidence(cls_logits, id2label, classifier_top_k):
    cls_scores = classifier_top_label_scores(cls_logits, id2label, classifier_top_k)
    return {
        label: {
            "score": score,
            "parts": {"cls": score},
        }
        for label, score in cls_scores.items()
    }


def decide_main_label_evidence(
    predictions_by_level,
    anchors_by_level,
    db,
    candidate_labels,
    cls_logits,
    id2label,
    classifier_top_k,
    margin_threshold,
):
    classifier_pred = id2label[str(int(cls_logits.argmax().item()))]
    old_label, old_reason = decide_main_label(anchors_by_level, classifier_pred)
    if not candidate_labels:
        return old_label, old_reason, {}

    evidence = score_candidate_labels(
        predictions_by_level,
        db,
        candidate_labels,
        cls_logits,
        id2label,
        classifier_top_k,
    )
    if not evidence:
        return old_label, old_reason, {}

    ranked = sorted(evidence.items(), key=lambda item: item[1]["score"], reverse=True)
    best_label, best_payload = ranked[0]
    reason = "candidate_evidence"
    if len(ranked) > 1:
        margin = best_payload["score"] - ranked[1][1]["score"]
        if margin < margin_threshold:
            return old_label, f"evidence_low_margin_{margin:.3f}_use_{old_reason}", evidence
        reason = f"candidate_evidence_margin_{margin:.3f}"
    return best_label, reason, evidence


@torch.no_grad()
def decide_main_label_reranker(reranker, predictions_by_level, db, candidate_labels, cls_logits, id2label, classifier_top_k):
    if reranker is None:
        return None, "", {}
    candidate_names, features = candidate_feature_tensor(
        predictions_by_level,
        db,
        candidate_labels,
        cls_logits,
        id2label,
        classifier_top_k,
    )
    if not candidate_names:
        return None, "", {}
    logits = reranker(features)
    probs = F.softmax(logits.float(), dim=0)
    best_index = int(probs.argmax().item())
    scores = {
        label: {
            "score": float(probs[i].item()),
            "parts": {"logit": float(logits[i].item())},
        }
        for i, label in enumerate(candidate_names)
    }
    return candidate_names[best_index], "reranker", scores


def select_label_anchors(anchors_by_level, main_label, prompt_top_k):
    selected = {"low": [], "mid": [], "high": []}
    for level, anchors in anchors_by_level.items():
        matched = [anchor for anchor in anchors if anchor["label"] == main_label]
        if matched:
            selected[level].extend(matched[:prompt_top_k])

    if not any(selected.values()) and anchors_by_level:
        fallback_level = "mid" if "mid" in anchors_by_level else next(iter(anchors_by_level))
        selected[fallback_level].append(anchors_by_level[fallback_level][0])
    return selected


def retrieve_label_anchors(predictions_by_level, db, main_label, prompt_top_k):
    if main_label not in db["label_indices"]:
        return None
    selected = {"low": [], "mid": [], "high": []}
    candidate_indices = db["label_indices"][main_label]
    for level, pred in predictions_by_level.items():
        query = F.normalize(pred.float(), dim=-1)
        scores = query @ db["embeddings"][level].t()
        label_scores = scores[:, candidate_indices]
        top_values, top_positions = torch.topk(
            label_scores,
            k=min(prompt_top_k, candidate_indices.numel()),
            dim=-1,
        )
        for score, position in zip(top_values[0], top_positions[0]):
            index = int(candidate_indices[int(position)].item())
            selected[level].append(format_anchor(db, index, float(score.item()), level))
    return selected


def select_decision_anchors(anchors_by_level, classifier_pred, prompt_top_k):
    main_label, reason = decide_main_label(anchors_by_level, classifier_pred)
    return select_label_anchors(anchors_by_level, main_label, prompt_top_k), main_label, reason


def select_prompt_anchors(anchors_by_level, anchor_mode, prompt_top_k):
    if anchor_mode == "top1_per_level":
        return {
            level: anchors[:1]
            for level, anchors in anchors_by_level.items()
        }
    if anchor_mode == "topk_per_level":
        return anchors_by_level
    if anchor_mode == "vote":
        return select_vote_anchors(anchors_by_level, prompt_top_k)

    flattened = []
    for level, anchors in anchors_by_level.items():
        for anchor in anchors:
            flattened.append((level, anchor))
    flattened.sort(key=lambda item: item[1]["score"], reverse=True)
    selected = {"low": [], "mid": [], "high": []}
    for level, anchor in flattened[:prompt_top_k]:
        selected[level].append(anchor)
    return selected


def format_evidence_scores(evidence, limit=5):
    if not evidence:
        return ""
    ranked = sorted(evidence.items(), key=lambda item: item[1]["score"], reverse=True)[:limit]
    return "|".join(f"{label}:{payload['score']:.3f}" for label, payload in ranked)


def build_retrieval_prompt(
    anchors_by_level,
    threshold,
    anchor_mode,
    prompt_top_k,
    classifier_pred=None,
    preselected_anchors=None,
    decision_label="",
    decision_reason="",
    generation_prompt_style="conservative",
    decoupled_semantics=False,
    level_thresholds=None,
    drop_unreliable_low=False,
):
    if preselected_anchors is not None:
        prompt_anchors = preselected_anchors
    elif anchor_mode == "decision":
        prompt_anchors, decision_label, decision_reason = select_decision_anchors(
            anchors_by_level,
            classifier_pred,
            prompt_top_k,
        )
    else:
        prompt_anchors = select_prompt_anchors(anchors_by_level, anchor_mode, prompt_top_k)
    reliable = []
    uncertain = []
    level_thresholds = level_thresholds or {}
    for level, anchors in prompt_anchors.items():
        for anchor in anchors:
            level_threshold = level_thresholds.get(level, threshold)
            if drop_unreliable_low and level == "low" and anchor["score"] < level_threshold:
                continue
            if decoupled_semantics and level in {"low", "high"}:
                item = f"{level} ({anchor['score']:.2f}) - {anchor['semantic_text']}"
            else:
                item = f"{level}: {anchor['label']} ({anchor['score']:.2f}) - {anchor['semantic_text']}"
            if anchor["score"] >= level_threshold:
                reliable.append(item)
            else:
                uncertain.append(item)
    if not reliable:
        flattened = [(level, anchor) for level, anchors in prompt_anchors.items() for anchor in anchors]
        if flattened:
            best_level, best_anchor = max(flattened, key=lambda pair: pair[1]["score"])
            if decoupled_semantics and best_level in {"low", "high"}:
                fallback_item = f"{best_level} ({best_anchor['score']:.2f}) - {best_anchor['semantic_text']}"
            else:
                fallback_item = (
                    f"{best_level}: {best_anchor['label']} ({best_anchor['score']:.2f}) - {best_anchor['semantic_text']}"
                )
            reliable.append(fallback_item)
            uncertain = [item for item in uncertain if item != fallback_item]
        else:
            reliable.append("none")

    reliable_text = "\n".join(f"- {item}" for item in reliable[:6])
    uncertain_text = "\n".join(f"- {item}" for item in uncertain[:6]) if uncertain else "- none"
    decision_text = ""
    if decision_label:
        decision_text = f"Main semantic decision: {decision_label} ({decision_reason}).\n\n"
    if generation_prompt_style == "structured":
        label_hint = decision_label or classifier_pred or "the main object"
        instruction = (
            "Instruction:\n"
            "Generate exactly one English sentence in this exact format:\n"
            f"The category is {label_hint}; it shows <a concise visual description>.\n"
            "Use the decided category as the main object and do not replace it with another object. "
            "After 'it shows', describe the main object, action or pose, visual attributes, "
            "and scene/context using only reliable cues. Do not mention scores, uncertainty, or retrieved cues. "
            "For low/high cues, use only their described attributes or scenes, not their source labels."
        )
    else:
        label_hint = decision_label or classifier_pred or "the main object"
        instruction = (
            "Instruction:\n"
            f"Generate one conservative English image caption about the main object category '{label_hint}'. "
            "Do not add details that are not supported. Output exactly one short sentence. "
            "For low/high cues, use only their described attributes or scenes, not their source labels."
        )
    return (
        decision_text
        + "Reliable semantic cues retrieved from EEG:\n"
        f"{reliable_text}\n\n"
        "Uncertain cues, do not rely on them unless necessary:\n"
        f"{uncertain_text}\n\n"
        + instruction
    )


@torch.no_grad()
def generate_from_prompt(tokenizer, llm, prompt, device, max_new_tokens):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You write concise and conservative image captions."},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[:, inputs["input_ids"].size(1):]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return clean_generated_caption(text)


def make_loader(paths, data_cfg, tokenizer, args):
    dataset = EEGCaptionDataset(
        eeg_dataset=paths.eeg_dataset,
        splits_path=paths.splits_path,
        image_dir=paths.image_dir,
        tokenizer=tokenizer,
        split_name="test",
        split_num=data_cfg.split_num,
        time_low=data_cfg.time_low,
        time_high=data_cfg.time_high,
        instruction=data_cfg.instruction,
        max_caption_tokens=data_cfg.max_caption_tokens,
        caption_map_path=args.caption_map_path,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_caption_batch(batch, tokenizer.pad_token_id),
    )


def main():
    args = parse_args()
    paths = configure_paths(args)
    data_cfg = DataConfig()
    ensure_source_on_path(paths.source_root)
    from constants import id2label

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        paths.llm_path,
        local_files_only=True,
        trust_remote_code=args.llm_trust_remote_code,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    test_loader = make_loader(paths, data_cfg, tokenizer, args)

    db = load_semantic_db(args.semantic_db_path, device)
    eeg_encoder, multi_head = load_retrieval_model(args, device)
    reranker = load_reranker(args.reranker_path, device)
    llm = None if args.skip_llm else load_llm_model(paths.llm_path, device, args.llm_trust_remote_code)

    rows = []
    seen = 0
    for batch in tqdm(test_loader, desc="Stage 4 retrieval infer"):
        if args.max_samples > 0 and seen >= args.max_samples:
            break
        eeg = batch["eeg"].unsqueeze(1).to(device)
        eeg_feat, cls_logits = eeg_encoder(eeg)
        pred_low, pred_mid, pred_high = multi_head(eeg_feat)
        batch_size = eeg.size(0)
        for i in range(batch_size):
            if args.max_samples > 0 and seen >= args.max_samples:
                break
            predictions_by_level = {
                "low": pred_low[i:i + 1],
                "mid": pred_mid[i:i + 1],
                "high": pred_high[i:i + 1],
            }
            if args.classifier_only:
                predictions_by_level = {}
            predictions_by_level = {
                level: pred
                for level, pred in predictions_by_level.items()
                if level in set(args.semantic_levels)
            }
            candidate_labels = set()
            candidate_mask = None
            if args.label_filter == "candidate":
                candidate_labels.update(classifier_top_labels(cls_logits[i], id2label, args.classifier_top_k))
                if args.decoupled_semantics:
                    label_predictions = {
                        level: pred
                        for level, pred in predictions_by_level.items()
                        if level == "mid"
                    }
                    candidate_labels.update(top_retrieval_labels(label_predictions, db, args.top_k))
                else:
                    candidate_labels.update(top_retrieval_labels(predictions_by_level, db, args.top_k))
                candidate_mask = make_candidate_mask(db, candidate_labels, device)

            anchors_by_level = {}
            top_labels = {"low": "", "mid": "", "high": ""}
            top_scores = {"low": "", "mid": "", "high": ""}
            for level, pred in predictions_by_level.items():
                if candidate_mask is not None:
                    scores, indices = retrieve_one_filtered(pred, db["embeddings"][level], candidate_mask, args.top_k)
                else:
                    scores, indices = retrieve_one(pred, db["embeddings"][level], args.top_k)
                anchors = [format_anchor(db, int(idx), float(score), level) for score, idx in zip(scores[0], indices[0])]
                anchors_by_level[level] = anchors
                top_labels[level] = anchors[0]["label"]
                top_scores[level] = anchors[0]["score"]

            predicted_label = id2label[str(int(cls_logits[i].argmax().item()))]
            main_label = ""
            decision_reason = ""
            decision_evidence = {}
            preselected_anchors = None
            if args.main_label_source == "classifier":
                main_label = predicted_label
                decision_reason = "classifier_main_label"
                decision_evidence = classifier_only_evidence(cls_logits[i], id2label, args.classifier_top_k)
                if args.classifier_only:
                    preselected_anchors = {}
                else:
                    preselected_anchors = retrieve_label_anchors(
                        predictions_by_level,
                        db,
                        main_label,
                        args.prompt_top_k,
                    )
                    if preselected_anchors is None:
                        preselected_anchors = select_prompt_anchors(
                            anchors_by_level,
                            "topk_per_level",
                            args.prompt_top_k,
                        )
            elif args.anchor_mode == "evidence" and args.decoupled_semantics:
                main_label = predicted_label
                decision_reason = "classifier_only" if args.classifier_only else "decoupled_classifier"
                decision_evidence = classifier_only_evidence(cls_logits[i], id2label, args.classifier_top_k)
                preselected_anchors = {} if args.classifier_only else select_prompt_anchors(
                    anchors_by_level,
                    "topk_per_level",
                    args.prompt_top_k,
                )
            elif args.anchor_mode == "evidence" and reranker is not None:
                rerank_label, rerank_reason, decision_evidence = decide_main_label_reranker(
                    reranker,
                    predictions_by_level,
                    db,
                    candidate_labels,
                    cls_logits[i],
                    id2label,
                    args.classifier_top_k,
                )
                if rerank_label:
                    main_label = rerank_label
                    decision_reason = rerank_reason
                    preselected_anchors = retrieve_label_anchors(
                        predictions_by_level,
                        db,
                        main_label,
                        args.prompt_top_k,
                    )
            elif args.anchor_mode == "evidence":
                main_label, decision_reason, decision_evidence = decide_main_label_evidence(
                    predictions_by_level,
                    anchors_by_level,
                    db,
                    candidate_labels,
                    cls_logits[i],
                    id2label,
                    args.classifier_top_k,
                    args.evidence_margin_threshold,
                )
                preselected_anchors = retrieve_label_anchors(
                    predictions_by_level,
                    db,
                    main_label,
                    args.prompt_top_k,
                )
            prompt = build_retrieval_prompt(
                anchors_by_level,
                args.reliable_threshold,
                args.anchor_mode,
                args.prompt_top_k,
                predicted_label,
                preselected_anchors=preselected_anchors,
                decision_label=main_label,
                decision_reason=decision_reason,
                generation_prompt_style=args.generation_prompt_style,
                decoupled_semantics=args.decoupled_semantics,
                level_thresholds={
                    "low": args.low_reliable_threshold if args.low_reliable_threshold is not None else args.reliable_threshold,
                    "mid": args.mid_reliable_threshold if args.mid_reliable_threshold is not None else args.reliable_threshold,
                    "high": args.high_reliable_threshold if args.high_reliable_threshold is not None else args.reliable_threshold,
                },
                drop_unreliable_low=args.drop_unreliable_low,
            )
            generated = "" if args.skip_llm else generate_from_prompt(tokenizer, llm, prompt, device, args.max_new_tokens)
            true_label = id2label[str(batch["labels"][i].item())]
            if args.anchor_mode == "decision":
                main_label, decision_reason = decide_main_label(anchors_by_level, predicted_label)
            elif args.anchor_mode != "evidence":
                main_label, decision_reason = "", ""
            rows.append({
                "image_name": batch["image_names"][i],
                "reference_caption": batch["captions"][i],
                "true_object": true_label,
                "classifier_pred": predicted_label,
                "classifier_correct": predicted_label == true_label,
                "top_low_label": top_labels["low"],
                "top_mid_label": top_labels["mid"],
                "top_high_label": top_labels["high"],
                "top_low_score": top_scores["low"],
                "top_mid_score": top_scores["mid"],
                "top_high_score": top_scores["high"],
                "semantic_levels": "|".join(args.semantic_levels),
                "decoupled_semantics": args.decoupled_semantics,
                "retrieval_label_hit_any": true_label in [label for label in top_labels.values() if label],
                "candidate_labels": "|".join(sorted(candidate_labels)) if candidate_labels else "",
                "candidate_label_hit": true_label in candidate_labels if candidate_labels else "",
                "main_label": main_label,
                "decision_reason": decision_reason,
                "decision_scores": format_evidence_scores(decision_evidence),
                "main_label_correct": main_label == true_label if main_label else "",
                "token_f1": token_f1(batch["captions"][i], generated) if generated else "",
                "generated_caption": generated,
                "prompt": prompt,
            })
            seen += 1

    if not rows:
        raise RuntimeError("No rows generated.")
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cls_acc = sum(row["classifier_correct"] for row in rows) / len(rows)
    hit_any = sum(row["retrieval_label_hit_any"] for row in rows) / len(rows)
    print(f"Saved retrieval generations to {args.output_csv}")
    print(f"Classifier accuracy: {cls_acc:.4f}")
    print(f"Retrieval label hit any level: {hit_any:.4f}")
    main_values = [row["main_label_correct"] for row in rows if row["main_label_correct"] != ""]
    if main_values:
        print(f"Main label accuracy: {sum(main_values) / len(main_values):.4f}")
    candidate_values = [row["candidate_label_hit"] for row in rows if row["candidate_label_hit"] != ""]
    if candidate_values:
        print(f"Candidate label hit: {sum(candidate_values) / len(candidate_values):.4f}")
    if not args.skip_llm:
        f1_values = [row["token_f1"] for row in rows if row["token_f1"] != ""]
        print(f"Average token F1: {sum(f1_values) / max(1, len(f1_values)):.4f}")
    print("\nFirst rows:")
    def fmt_score(value):
        return f"{value:.2f}" if isinstance(value, (int, float)) else ""
    for row in rows[:5]:
        print(f"- image={row['image_name']} true={row['true_object']} cls={row['classifier_pred']}")
        print(
            f"  low={row['top_low_label']}({fmt_score(row['top_low_score'])}) "
            f"mid={row['top_mid_label']}({fmt_score(row['top_mid_score'])}) "
            f"high={row['top_high_label']}({fmt_score(row['top_high_score'])})"
        )
        if row["generated_caption"]:
            print(f"  gen={row['generated_caption']}")


if __name__ == "__main__":
    main()
