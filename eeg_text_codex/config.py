from dataclasses import dataclass


@dataclass
class PathConfig:
    """All filesystem paths used by the main route.

    The source project is kept as a dependency because it already contains the
    trained EEG encoder, data files, CLIP processor files, and local LLM.
    """

    source_root: str = "/home/dell/桌面/coding/2025/hl/LLM_text"
    eeg_dataset: str = "/home/dell/桌面/coding/2025/hl/LLM_text/data_copy/block/eeg_55_95_std.pth"
    splits_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/data_copy/block/block_splits_by_image_all.pth"
    image_dir: str = "/home/dell/桌面/coding/2025/hl/LLM_text/data_copy/images"
    eeg_encoder_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/eeg_encoder_55-95_40_classes"
    eeg_encoder_type: str = "channelnet"
    eeg_feature_dim: int = 512
    num_classes: int = 40
    eegpt_model_dir: str = ""
    eegpt_checkpoint_path: str = ""
    eegpt_import: str = "archs.EEGPT_mcae:EEGTransformer"
    eegpt_model_kwargs: str = ""
    eegpt_backbone_out_dim: int = 512
    eegpt_input_channels: int = 128
    eegpt_channels: int = 58
    eegpt_target_time_len: int = 1024
    multi_head_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/eeg_head/best_multi_head.pth"
    llm_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/llm/llm"
    output_dir: str = "/home/dell/桌面/coding/2025/hl/LLM_text/eeg_head/moe_projector_codex"
    staged_output_dir: str = "/home/dell/桌面/coding/2025/hl/eeg-text-codex/outputs"
    clip_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/llm/clip"
    qwen_vl_path: str = "/home/dell/桌面/coding/2025/hl/LLM_text/llm/Qwen-VL-Chat"
    structured_caption_path: str = "/home/dell/桌面/coding/2025/hl/eeg-text-codex/outputs/qwen_structured_captions.json"


@dataclass
class DataConfig:
    """Dataset and prompt settings."""

    split_num: int = 0
    subject: int = 0
    time_low: int = 20
    time_high: int = 460
    max_caption_tokens: int = 80
    instruction: str = "Describe this image in one sentence:"
    use_pred_label: bool = True


@dataclass
class TrainConfig:
    """Optimization settings for training only MOE + projector."""

    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = -1
    save_steps: int = 500
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    grad_clip: float = 1.0
