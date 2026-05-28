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
from eeg_text_codex.modules import MultiHead
from eeg_text_codex.utils import ensure_source_on_path, get_device, load_eeg_encoder, set_seed


EEGPT_KWARGS = '{"img_size":[58,1024],"patch_size":64,"patch_stride":64,"embed_num":4,"embed_dim":512,"depth":8,"num_heads":8,"mlp_ratio":4.0,"qkv_bias":true}'


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 structured: align EEG heads to explicit low/mid/high text semantics.")
    parser.add_argument("--target_db_path", default=os.path.join(PathConfig.staged_output_dir, "structured_semantic_targets_all.pt"))
    parser.add_argument("--eeg_encoder_type", choices=["channelnet", "eegpt"], default="eegpt")
    parser.add_argument("--eeg_encoder_path", default=os.path.join(PathConfig.staged_output_dir, "stage1_eegpt_adapter", "best"))
    parser.add_argument("--eegpt_model_dir", default="external/EEGPT/downstream")
    parser.add_argument("--eegpt_checkpoint_path", default="external/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt")
    parser.add_argument("--eegpt_import", default="Modules.models.EEGPT_mcae:EEGTransformer")
    parser.add_argument("--eegpt_model_kwargs", default=EEGPT_KWARGS)
    parser.add_argument("--eegpt_backbone_out_dim", type=int, default=2048)
    parser.add_argument("--eeg_feature_dim", type=int, default=PathConfig.eeg_feature_dim)
    parser.add_argument("--output_dir", default=os.path.join(PathConfig.staged_output_dir, "stage2_eegpt_structured"))
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cls_loss_weight", type=float, default=0.1)
    parser.add_argument("--full_loss_weight", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def configure_paths(args):
    paths = PathConfig()
    paths.eeg_encoder_type = args.eeg_encoder_type
    paths.eeg_encoder_path = args.eeg_encoder_path
    paths.eegpt_model_dir = args.eegpt_model_dir
    paths.eegpt_checkpoint_path = args.eegpt_checkpoint_path
    paths.eegpt_import = args.eegpt_import
    paths.eegpt_model_kwargs = args.eegpt_model_kwargs
    paths.eegpt_backbone_out_dim = args.eegpt_backbone_out_dim
    paths.eeg_feature_dim = args.eeg_feature_dim
    return paths


def make_loader(paths, data_cfg, split_name, batch_size, shuffle, num_workers):
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


def load_target_db(path, device):
    db = torch.load(path, map_location="cpu")
    image_to_index = {name: i for i, name in enumerate(db["image_names"])}
    embeddings = {
        key: F.normalize(value.float(), dim=-1).to(device)
        for key, value in db["embeddings"].items()
    }
    return db, image_to_index, embeddings


def batch_targets(image_names, image_to_index, embeddings):
    indices = []
    missing = []
    for image_name in image_names:
        if image_name not in image_to_index:
            missing.append(image_name)
        else:
            indices.append(image_to_index[image_name])
    if missing:
        raise KeyError(f"Missing {len(missing)} images in structured target DB, first={missing[0]}")
    index_tensor = torch.tensor(indices, device=embeddings["low"].device, dtype=torch.long)
    return embeddings["low"][index_tensor], embeddings["mid"][index_tensor], embeddings["high"][index_tensor], embeddings["full"][index_tensor]


def contrastive_loss(pred, target, temperature):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def retrieval_top1(pred, target):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    labels = torch.arange(pred.size(0), device=pred.device)
    return ((pred @ target.t()).argmax(dim=1) == labels).float().mean().item()


def trainable_params(eeg_encoder, multi_head, encoder_type):
    params = list(multi_head.parameters())
    if encoder_type == "eegpt":
        params += eeg_encoder.trainable_parameters
    return params


def forward_batch(eeg_encoder, multi_head, batch, image_to_index, embeddings, device):
    eeg = batch["eeg"].unsqueeze(1).to(device)
    labels = batch["labels"].to(device)
    targets = batch_targets(batch["image_names"], image_to_index, embeddings)
    eeg_feat, cls_logits = eeg_encoder(eeg)
    predictions = multi_head(eeg_feat)
    return predictions, targets, cls_logits, labels


def compute_loss(predictions, targets, cls_logits, labels, args, cls_criterion):
    low_loss = contrastive_loss(predictions[0], targets[0], args.temperature)
    mid_loss = contrastive_loss(predictions[1], targets[1], args.temperature)
    high_loss = contrastive_loss(predictions[2], targets[2], args.temperature)
    full_loss = contrastive_loss(predictions[2], targets[3], args.temperature)
    cls_loss = cls_criterion(cls_logits, labels)
    loss = low_loss + mid_loss + high_loss + args.full_loss_weight * full_loss + args.cls_loss_weight * cls_loss
    return loss, {
        "low": low_loss,
        "mid": mid_loss,
        "high": high_loss,
        "full": full_loss,
        "cls": cls_loss,
    }


@torch.no_grad()
def evaluate(eeg_encoder, multi_head, loader, image_to_index, embeddings, args, device):
    eeg_encoder.eval()
    multi_head.eval()
    cls_criterion = nn.CrossEntropyLoss()
    totals = {"loss": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "full": 0.0, "cls": 0.0, "acc": 0.0, "r_low": 0.0, "r_mid": 0.0, "r_high": 0.0, "r_full": 0.0}
    count = 0
    for batch_id, batch in enumerate(tqdm(loader, desc="Stage 2 structured val", leave=False), start=1):
        if args.max_eval_batches > 0 and batch_id > args.max_eval_batches:
            break
        predictions, targets, cls_logits, labels = forward_batch(eeg_encoder, multi_head, batch, image_to_index, embeddings, device)
        loss, losses = compute_loss(predictions, targets, cls_logits, labels, args, cls_criterion)
        totals["loss"] += loss.item()
        for key, value in losses.items():
            totals[key] += value.item()
        totals["acc"] += (cls_logits.argmax(dim=1) == labels).float().mean().item()
        totals["r_low"] += retrieval_top1(predictions[0], targets[0])
        totals["r_mid"] += retrieval_top1(predictions[1], targets[1])
        totals["r_high"] += retrieval_top1(predictions[2], targets[2])
        totals["r_full"] += retrieval_top1(predictions[2], targets[3])
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(output_dir, eeg_encoder, multi_head, metadata, encoder_type):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(multi_head.state_dict(), os.path.join(output_dir, "multi_head.pt"))
    if encoder_type == "eegpt":
        eeg_encoder.save_adapter(os.path.join(output_dir, "eeg_encoder"), metadata)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    paths = configure_paths(args)
    data_cfg = DataConfig()
    ensure_source_on_path(paths.source_root)
    set_seed(args.seed)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    _, image_to_index, embeddings = load_target_db(args.target_db_path, device)

    train_loader = make_loader(paths, data_cfg, "train", args.batch_size, True, args.num_workers)
    val_loader = make_loader(paths, data_cfg, "val", args.batch_size, False, args.num_workers)
    test_loader = make_loader(paths, data_cfg, "test", args.batch_size, False, args.num_workers)

    eeg_encoder = load_eeg_encoder(paths, device)
    multi_head = MultiHead(args.eeg_feature_dim, 512).to(device)
    optimizer = torch.optim.AdamW(trainable_params(eeg_encoder, multi_head, args.eeg_encoder_type), lr=args.learning_rate, weight_decay=args.weight_decay)
    cls_criterion = nn.CrossEntropyLoss()
    best_val = float("inf")
    global_step = 0

    for epoch in range(args.num_epochs):
        eeg_encoder.train()
        if args.eeg_encoder_type == "eegpt":
            eeg_encoder.backbone.eval()
        multi_head.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 2 structured epoch {epoch + 1}/{args.num_epochs}")
        for local_step, batch in enumerate(pbar, start=1):
            predictions, targets, cls_logits, labels = forward_batch(eeg_encoder, multi_head, batch, image_to_index, embeddings, device)
            loss, losses = compute_loss(predictions, targets, cls_logits, labels, args, cls_criterion)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params(eeg_encoder, multi_head, args.eeg_encoder_type), 1.0)
            optimizer.step()
            global_step += 1
            running += loss.item()
            pbar.set_postfix({
                "loss": f"{running / local_step:.4f}",
                "low": f"{losses['low'].item():.3f}",
                "mid": f"{losses['mid'].item():.3f}",
                "high": f"{losses['high'].item():.3f}",
                "acc": f"{(cls_logits.argmax(dim=1) == labels).float().mean().item():.3f}",
            })
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        val_metrics = evaluate(eeg_encoder, multi_head, val_loader, image_to_index, embeddings, args, device)
        print(
            f"Epoch {epoch + 1}: val_loss={val_metrics['loss']:.6f} "
            f"low={val_metrics['low']:.4f} mid={val_metrics['mid']:.4f} high={val_metrics['high']:.4f} "
            f"full={val_metrics['full']:.4f} cls={val_metrics['cls']:.4f} acc={val_metrics['acc']:.4f} "
            f"r_low={val_metrics['r_low']:.4f} r_mid={val_metrics['r_mid']:.4f} "
            f"r_high={val_metrics['r_high']:.4f} r_full={val_metrics['r_full']:.4f}"
        )
        metadata = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "val_metrics": val_metrics,
            "target_db_path": args.target_db_path,
            "eeg_encoder_type": args.eeg_encoder_type,
            "temperature": args.temperature,
            "cls_loss_weight": args.cls_loss_weight,
            "full_loss_weight": args.full_loss_weight,
        }
        save_checkpoint(os.path.join(args.output_dir, "last"), eeg_encoder, multi_head, metadata, args.eeg_encoder_type)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(os.path.join(args.output_dir, "best"), eeg_encoder, multi_head, metadata, args.eeg_encoder_type)
            print(f"Saved best structured model to {os.path.join(args.output_dir, 'best')}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    test_metrics = evaluate(eeg_encoder, multi_head, test_loader, image_to_index, embeddings, args, device)
    print(
        f"Stage 2 structured test_loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.4f} "
        f"r_low={test_metrics['r_low']:.4f} r_mid={test_metrics['r_mid']:.4f} "
        f"r_high={test_metrics['r_high']:.4f} r_full={test_metrics['r_full']:.4f}"
    )


if __name__ == "__main__":
    main()
