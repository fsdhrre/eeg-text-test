"""可选阶段五：训练候选类别 reranker。

这个脚本属于消融/扩展实验，不是默认最终主线。它会冻结已经训练好的 EEG encoder 和
三层语义头，然后用“分类器 top-k + 检索 top-k”构造候选类别集合，再训练一个小型
MLP reranker，从候选类别的 evidence feature 中选择真实类别。

当前实验中，规则式 evidence decision 表现更好；保留该脚本是为了让 learned reranker
这个替代方案可复现。
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig, TrainConfig
from eeg_text_codex.data import EEGImageDataset, collate_eeg_label_batch
from eeg_text_codex.modules import CandidateReranker, MultiHead
from eeg_text_codex.rerank import FEATURE_NAMES, candidate_feature_tensor, ensure_label_indices
from eeg_text_codex.utils import ensure_source_on_path, get_device, load_eeg_encoder, set_seed
from scripts.stage4_retrieval_infer import (
    EEGPT_KWARGS,
    configure_paths,
    load_semantic_db,
    retrieve_one,
    top_retrieval_labels,
)


def parse_args():
    """定义 reranker、冻结 checkpoint 和候选类别生成相关参数。"""

    parser = argparse.ArgumentParser(description="Stage 5: train a candidate label reranker on retrieval evidence.")
    parser.add_argument("--checkpoint_dir", default=os.path.join(PathConfig.staged_output_dir, "stage2_eegpt_structured", "best"))
    parser.add_argument("--semantic_db_path", default=os.path.join(PathConfig.staged_output_dir, "structured_semantic_targets_all_smoke.pt"))
    parser.add_argument("--output_dir", default=os.path.join(PathConfig.staged_output_dir, "stage5_candidate_reranker"))
    parser.add_argument("--eeg_encoder_type", choices=["channelnet", "eegpt"], default="eegpt")
    parser.add_argument("--eegpt_model_dir", default="external/EEGPT/downstream")
    parser.add_argument("--eegpt_checkpoint_path", default="external/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt")
    parser.add_argument("--eegpt_import", default="Modules.models.EEGPT_mcae:EEGTransformer")
    parser.add_argument("--eegpt_model_kwargs", default=EEGPT_KWARGS)
    parser.add_argument("--eegpt_backbone_out_dim", type=int, default=2048)
    parser.add_argument("--eeg_feature_dim", type=int, default=PathConfig.eeg_feature_dim)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--classifier_top_k", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--add_gold_to_train_candidates", action="store_true", default=True)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def make_loader(paths, data_cfg, split_name, batch_size, shuffle, num_workers):
    """为 reranker 训练/评估构建只读取 EEG 和标签的 loader。"""

    dataset = EEGImageDataset(
        eeg_dataset=paths.eeg_dataset,
        splits_path=paths.splits_path,
        image_dir=paths.image_dir,
        split_name=split_name,
        split_num=data_cfg.split_num,
        time_low=data_cfg.time_low,
        time_high=data_cfg.time_high,
        load_image=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_eeg_label_batch,
    )


def classifier_top_labels(cls_logits, id2label, top_k):
    """把分类器 logits 转成 top-k 类别名称集合。"""

    _, indices = torch.topk(cls_logits.float(), k=min(top_k, cls_logits.numel()), dim=-1)
    return {id2label[str(int(index.item()))] for index in indices}


@torch.no_grad()
def build_candidate_example(eeg_encoder, multi_head, batch, db, id2label, args, device, add_gold=False):
    """为一个 EEG batch 构造 reranker 训练样本。

    每个样本对应一个长度可变的候选类别列表。对每个候选类别，
    ``candidate_feature_tensor`` 会构造一个 feature vector，汇总分类器置信度以及
    low / mid / high 三层检索证据。target 是真实类别在候选列表中的下标。
    """

    eeg = batch["eeg"].unsqueeze(1).to(device)
    labels = batch["labels"].to(device)
    eeg_feat, cls_logits = eeg_encoder(eeg)
    pred_low, pred_mid, pred_high = multi_head(eeg_feat)

    examples = []
    for i in range(eeg.size(0)):
        predictions_by_level = {
            "low": pred_low[i:i + 1],
            "mid": pred_mid[i:i + 1],
            "high": pred_high[i:i + 1],
        }
        true_label = id2label[str(int(labels[i].item()))]
        candidate_labels = set()
        candidate_labels.update(classifier_top_labels(cls_logits[i], id2label, args.classifier_top_k))
        candidate_labels.update(top_retrieval_labels(predictions_by_level, db, args.top_k))
        if add_gold:
            # 训练时加入 gold label，可以避免“分类器和检索都没召回真值”时样本被丢掉。
            # 评估时关闭该选项，这样 candidate_hit 才是真实召回率。
            candidate_labels.add(true_label)

        candidate_names, features = candidate_feature_tensor(
            predictions_by_level,
            db,
            candidate_labels,
            cls_logits[i],
            id2label,
            args.classifier_top_k,
        )
        if true_label not in candidate_names:
            continue
        target = candidate_names.index(true_label)
        examples.append((features, torch.tensor(target, device=device, dtype=torch.long), candidate_names, true_label))
    return examples


def train_epoch(reranker, optimizer, eeg_encoder, multi_head, loader, db, id2label, args, device):
    """在长度可变的候选集合上训练 reranker 一个 epoch。"""

    reranker.train()
    total_loss = 0.0
    total = 0
    correct = 0
    skipped = 0
    for batch_id, batch in enumerate(tqdm(loader, desc="Stage 5 train", leave=False), start=1):
        if args.max_train_batches > 0 and batch_id > args.max_train_batches:
            break
        examples = build_candidate_example(
            eeg_encoder,
            multi_head,
            batch,
            db,
            id2label,
            args,
            device,
            add_gold=args.add_gold_to_train_candidates,
        )
        for features, target, _, _ in examples:
            logits = reranker(features)
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += 1
            correct += int(logits.argmax().item() == target.item())
        skipped += max(0, batch["eeg"].size(0) - len(examples))
    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "count": total,
        "skipped": skipped,
    }


@torch.no_grad()
def evaluate(reranker, eeg_encoder, multi_head, loader, db, id2label, args, device):
    """评估 reranker accuracy 以及候选集合是否召回真实类别。"""

    reranker.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    candidate_hit = 0
    skipped = 0
    for batch_id, batch in enumerate(tqdm(loader, desc="Stage 5 eval", leave=False), start=1):
        if args.max_eval_batches > 0 and batch_id > args.max_eval_batches:
            break
        examples = build_candidate_example(
            eeg_encoder,
            multi_head,
            batch,
            db,
            id2label,
            args,
            device,
            add_gold=False,
        )
        for features, target, _, _ in examples:
            logits = reranker(features)
            total_loss += F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)).item()
            total += 1
            candidate_hit += 1
            correct += int(logits.argmax().item() == target.item())
        skipped += max(0, batch["eeg"].size(0) - len(examples))
    total_seen = total + skipped
    return {
        "loss": total_loss / max(1, total),
        "acc_on_hit": correct / max(1, total),
        "acc_all": correct / max(1, total_seen),
        "candidate_hit": candidate_hit / max(1, total_seen),
        "count": total,
        "skipped": skipped,
    }


def save_checkpoint(path, reranker, metadata):
    """保存 reranker 权重以及 feature 元信息。"""

    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "state_dict": reranker.state_dict(),
            "feature_names": FEATURE_NAMES,
            "metadata": metadata,
        },
        os.path.join(path, "reranker.pt"),
    )
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_source_on_path(PathConfig.source_root)
    from constants import id2label

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = configure_paths(args)
    data_cfg = DataConfig()
    # 加载和阶段四一致的语义数据库，保证 reranker feature 和推理阶段 evidence 一致。
    db = load_semantic_db(args.semantic_db_path, device)
    ensure_label_indices(db, device)

    train_loader = make_loader(paths, data_cfg, "train", args.batch_size, True, args.num_workers)
    val_loader = make_loader(paths, data_cfg, "val", args.batch_size, False, args.num_workers)
    test_loader = make_loader(paths, data_cfg, "test", args.batch_size, False, args.num_workers)

    # 阶段五不更新 EEG 模型和语义头，只学习如何根据已有 evidence feature 给候选类别打分。
    eeg_encoder = load_eeg_encoder(paths, device).eval()
    multi_head = MultiHead(args.eeg_feature_dim, 512).to(device).eval()
    multi_head.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "multi_head.pt"), map_location=device))
    for parameter in eeg_encoder.parameters():
        parameter.requires_grad = False
    for parameter in multi_head.parameters():
        parameter.requires_grad = False

    # CandidateReranker 是一个小型 MLP，独立作用于每个候选类别；
    # 最后在候选类别维度做 softmax，选出最终类别。
    reranker = CandidateReranker(len(FEATURE_NAMES), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val = -1.0

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(reranker, optimizer, eeg_encoder, multi_head, train_loader, db, id2label, args, device)
        val_metrics = evaluate(reranker, eeg_encoder, multi_head, val_loader, db, id2label, args, device)
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_acc_all={val_metrics['acc_all']:.4f} val_acc_on_hit={val_metrics['acc_on_hit']:.4f} "
            f"val_candidate_hit={val_metrics['candidate_hit']:.4f}"
        )
        metadata = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "checkpoint_dir": args.checkpoint_dir,
            "semantic_db_path": args.semantic_db_path,
            "feature_names": FEATURE_NAMES,
            "classifier_top_k": args.classifier_top_k,
            "top_k": args.top_k,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        save_checkpoint(os.path.join(args.output_dir, "last"), reranker, metadata)
        if val_metrics["acc_all"] > best_val:
            best_val = val_metrics["acc_all"]
            save_checkpoint(os.path.join(args.output_dir, "best"), reranker, metadata)
            print(f"Saved best reranker to {os.path.join(args.output_dir, 'best')}")

    test_metrics = evaluate(reranker, eeg_encoder, multi_head, test_loader, db, id2label, args, device)
    print(
        f"Stage 5 test_acc_all={test_metrics['acc_all']:.4f} "
        f"test_acc_on_hit={test_metrics['acc_on_hit']:.4f} "
        f"test_candidate_hit={test_metrics['candidate_hit']:.4f}"
    )


if __name__ == "__main__":
    main()
