"""阶段一：训练 EEG 侧的 adapter / 分类头。

这是整条干净流程里的第一个监督训练脚本。它会读取 ``PathConfig`` 中指定的
EEG 编码器；推荐设置下使用预训练 EEGPT backbone，并在外面接一个轻量 adapter。
EEGPT backbone 会在 ``load_eeg_encoder`` 中被冻结，训练时只更新 adapter 和分类头。

该阶段输出的 checkpoint 会被阶段二继续使用，用来把同一个 EEG 特征提取器对齐到
low / mid / high 三种语义文本嵌入空间。
"""

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
    """定义阶段一分类预训练所需的命令行参数。"""

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
    """为指定 split 构建只读取 EEG 和标签的 DataLoader。

    阶段一只需要 EEG 张量和类别标签，所以这里不加载图片。数据集中仍会保留
    image_name，方便后续阶段把每个 EEG 样本对应到语义监督目标。
    """

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
    """在不更新参数的情况下计算分类 loss 和 accuracy。"""

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

    # 兼容旧命令：如果用户选择 EEGPT，但 output_dir 仍是历史 ChannelNet 路径，
    # 自动切到 EEGPT 专用目录，避免两类 checkpoint 混在一起。
    default_channelnet_output = os.path.join(PathConfig.staged_output_dir, "stage1_channelnet")
    if args.eeg_encoder_type == "eegpt" and args.output_dir == default_channelnet_output:
        args.output_dir = os.path.join(PathConfig.staged_output_dir, "stage1_eegpt_adapter")
    # PathConfig 保存本地数据和模型路径；命令行覆盖项会在构建 encoder 前写回这里。
    paths = PathConfig()
    data_cfg = DataConfig()
    ensure_source_on_path(paths.source_root)
    set_seed(args.seed)

    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # train / val / test 三个 loader 使用相同 EEG 时间窗和标签映射。
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

    # 对 EEGPT 来说，load_eeg_encoder 返回一个 wrapper：backbone 已冻结，
    # trainable_parameters 里只包含 adapter / 分类头参数。
    model = load_eeg_encoder(paths, device)
    model.train()

    # EEGPT 分支只训练 adapter / 分类头；ChannelNet 分支仅作为旧实验兼容路径保留。
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
            # 数据集中的 EEG 形状是 [B, C, T]；encoder 需要显式增加一个类似图像通道的维度，
            # 变成 [B, 1, C, T]。
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

        # 每个 epoch 都保存 last，便于恢复/调试；只有验证集 accuracy 提升时才保存 best。
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

    # 最终测试前重新加载 best checkpoint，保证 test 指标对应的是验证集选出的模型。
    if args.eeg_encoder_type == "eegpt":
        paths.eeg_encoder_path = os.path.join(args.output_dir, "best")
        best_model = load_eeg_encoder(paths, device)
    else:
        best_model = load_channelnet_encoder(os.path.join(args.output_dir, "best"), device)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device, "Stage 1 test", args.max_eval_batches)
    print(f"Stage 1 test_loss={test_loss:.6f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
