import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import DataConfig, PathConfig, TrainConfig
from eeg_text_codex.data import EEGImageDataset, collate_eeg_label_batch
from eeg_text_codex.utils import ensure_source_on_path, get_device, load_channelnet_encoder, load_eeg_encoder, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: train EEG encoder adapter for classification.")
    parser.add_argument("--eeg_encoder_type", choices=["channelnet", "eegpt"], default=PathConfig.eeg_encoder_type)
    parser.add_argument("--init_encoder_path", default=PathConfig.eeg_encoder_path)
    parser.add_argument("--eegpt_model_dir", default=PathConfig.eegpt_model_dir)
    parser.add_argument("--eegpt_checkpoint_path", default=PathConfig.eegpt_checkpoint_path)
    parser.add_argument("--eegpt_import", default=PathConfig.eegpt_import)
    parser.add_argument("--eegpt_model_kwargs", default=PathConfig.eegpt_model_kwargs)
    parser.add_argument("--eegpt_backbone_out_dim", type=int, default=PathConfig.eegpt_backbone_out_dim)
    parser.add_argument("--eegpt_input_channels", type=int, default=PathConfig.eegpt_input_channels)
    parser.add_argument("--eegpt_channels", type=int, default=PathConfig.eegpt_channels)
    parser.add_argument("--eegpt_target_time_len", type=int, default=PathConfig.eegpt_target_time_len)
    parser.add_argument("--eeg_feature_dim", type=int, default=PathConfig.eeg_feature_dim)
    parser.add_argument("--output_dir", default=os.path.join(PathConfig.staged_output_dir, "stage1_channelnet"))
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


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


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc, max_batches=-1):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_id, batch in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
        if max_batches > 0 and batch_id > max_batches:
            break
        eeg = batch["eeg"].unsqueeze(1).to(device)
        labels = batch["labels"].to(device)
        _, logits = model(eeg)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def main():
    args = parse_args()
    default_channelnet_output = os.path.join(PathConfig.staged_output_dir, "stage1_channelnet")
    if args.eeg_encoder_type == "eegpt" and args.output_dir == default_channelnet_output:
        args.output_dir = os.path.join(PathConfig.staged_output_dir, "stage1_eegpt_adapter")
    paths = PathConfig()
    data_cfg = DataConfig()
    ensure_source_on_path(paths.source_root)
    set_seed(args.seed)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader = make_loader(paths, data_cfg, "train", args.batch_size, True, args.num_workers)
    val_loader = make_loader(paths, data_cfg, "val", args.batch_size, False, args.num_workers)
    test_loader = make_loader(paths, data_cfg, "test", args.batch_size, False, args.num_workers)

    paths.eeg_encoder_type = args.eeg_encoder_type
    paths.eeg_encoder_path = args.init_encoder_path
    paths.eegpt_model_dir = args.eegpt_model_dir
    paths.eegpt_checkpoint_path = args.eegpt_checkpoint_path
    paths.eegpt_import = args.eegpt_import
    paths.eegpt_model_kwargs = args.eegpt_model_kwargs
    paths.eegpt_backbone_out_dim = args.eegpt_backbone_out_dim
    paths.eegpt_input_channels = args.eegpt_input_channels
    paths.eegpt_channels = args.eegpt_channels
    paths.eegpt_target_time_len = args.eegpt_target_time_len
    paths.eeg_feature_dim = args.eeg_feature_dim

    model = load_eeg_encoder(paths, device)
    model.train()

    if args.eeg_encoder_type == "eegpt":
        trainable_params = model.trainable_parameters
    else:
        trainable_params = model.parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        pbar = tqdm(train_loader, desc=f"Stage 1 train epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            eeg = batch["eeg"].unsqueeze(1).to(device)
            labels = batch["labels"].to(device)
            _, logits = model(eeg)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_count += labels.size(0)
            pbar.set_postfix({
                "loss": f"{running_loss / max(1, running_count):.4f}",
                "acc": f"{running_correct / max(1, running_count):.4f}",
            })

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "Stage 1 val", args.max_eval_batches)
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        if args.eeg_encoder_type == "eegpt":
            model.save_adapter(
                os.path.join(args.output_dir, "last"),
                {
                    "encoder_type": "eegpt",
                    "eegpt_model_dir": args.eegpt_model_dir,
                    "eegpt_checkpoint_path": args.eegpt_checkpoint_path,
                    "eegpt_import": args.eegpt_import,
                    "eegpt_model_kwargs": args.eegpt_model_kwargs,
                    "eegpt_backbone_out_dim": args.eegpt_backbone_out_dim,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                },
            )
        else:
            model.save_pretrained(os.path.join(args.output_dir, "last"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.eeg_encoder_type == "eegpt":
                model.save_adapter(
                    os.path.join(args.output_dir, "best"),
                    {
                        "encoder_type": "eegpt",
                        "eegpt_model_dir": args.eegpt_model_dir,
                        "eegpt_checkpoint_path": args.eegpt_checkpoint_path,
                        "eegpt_import": args.eegpt_import,
                        "eegpt_model_kwargs": args.eegpt_model_kwargs,
                        "eegpt_backbone_out_dim": args.eegpt_backbone_out_dim,
                        "val_acc": best_val_acc,
                        "epoch": epoch + 1,
                    },
                )
            else:
                model.save_pretrained(os.path.join(args.output_dir, "best"))
            print(f"Saved best EEG encoder to {os.path.join(args.output_dir, 'best')} (val_acc={best_val_acc:.4f})")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if args.eeg_encoder_type == "eegpt":
        paths.eeg_encoder_path = os.path.join(args.output_dir, "best")
        best_model = load_eeg_encoder(paths, device)
    else:
        best_model = load_channelnet_encoder(os.path.join(args.output_dir, "best"), device)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device, "Stage 1 test", args.max_eval_batches)
    print(f"Stage 1 test_loss={test_loss:.6f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
