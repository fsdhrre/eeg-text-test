import json
import os
import random
import sys
import types
from typing import Any, Dict

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from .modules import EEGPTAdapterClassifier, build_eegpt_backbone, load_eegpt_model_kwargs


def ensure_source_on_path(source_root: str) -> None:
    """Make the original project importable from this standalone folder."""

    for path in [source_root, os.path.join(source_root, "model_use")]:
        if path not in sys.path:
            sys.path.append(path)


def set_seed(seed: int) -> None:
    """Set common random seeds for repeatable training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    """Return CUDA when requested and available, otherwise CPU."""

    return torch.device(device_name if torch.cuda.is_available() else "cpu")


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze a module and switch it to eval mode."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def load_channelnet_encoder(model_dir: str, device: torch.device):
    """Load the trained ChannelNet EEG encoder from config + safetensors.

    The local ChannelNet class does not define `config_class`, so vanilla
    `from_pretrained` cannot load it reliably. Loading config and state dict
    explicitly keeps the dependency clear.
    """

    from llm.channelnet.config import EEGModelConfig
    from llm.channelnet.model import ChannelNetModel

    config_path = os.path.join(model_dir, "config.json")
    weight_path = os.path.join(model_dir, "model.safetensors")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict.pop("model_type", None)
    config_dict.pop("transformers_version", None)

    eeg_config = EEGModelConfig(**config_dict)
    encoder = ChannelNetModel(config=eeg_config)
    state_dict = load_file(weight_path, device=str(device))
    encoder.load_state_dict(state_dict, strict=False)
    return encoder.to(device)


def load_eegpt_encoder(
    model_dir: str,
    checkpoint_path: str,
    import_spec: str,
    model_kwargs_json: str,
    backbone_out_dim: int,
    device: torch.device,
    adapter_checkpoint_dir: str = "",
    adapter_dim: int = 512,
    num_classes: int = 40,
    input_channels: int = 128,
    eegpt_channels: int = 58,
    target_time_len: int = 1024,
) -> EEGPTAdapterClassifier:
    """Load frozen EEGPT plus trainable/loadable adapter classifier."""

    if not model_dir:
        raise ValueError("EEGPT model_dir is required. Pass --eegpt_model_dir.")
    if not import_spec:
        raise ValueError("EEGPT import spec is required. Pass --eegpt_import, e.g. archs.EEGPT_mcae:EEGTransformer.")
    model_kwargs = load_eegpt_model_kwargs(model_kwargs_json)
    backbone = build_eegpt_backbone(
        model_dir=model_dir,
        import_spec=import_spec,
        checkpoint_path=checkpoint_path,
        model_kwargs=model_kwargs,
    )
    encoder = EEGPTAdapterClassifier(
        backbone=backbone,
        backbone_out_dim=backbone_out_dim,
        adapter_dim=adapter_dim,
        num_classes=num_classes,
        input_channels=input_channels,
        eegpt_channels=eegpt_channels,
        target_time_len=target_time_len,
    ).to(device)
    if adapter_checkpoint_dir and os.path.exists(os.path.join(adapter_checkpoint_dir, "encoder.pt")):
        encoder.load_adapter(adapter_checkpoint_dir, device)
    return encoder


def load_eeg_encoder(paths, device: torch.device):
    """Load the configured EEG encoder behind the common `(feature, logits)` API."""

    encoder_type = getattr(paths, "eeg_encoder_type", "channelnet")
    if encoder_type == "channelnet":
        return load_channelnet_encoder(paths.eeg_encoder_path, device)
    if encoder_type == "eegpt":
        return load_eegpt_encoder(
            model_dir=getattr(paths, "eegpt_model_dir", ""),
            checkpoint_path=getattr(paths, "eegpt_checkpoint_path", ""),
            import_spec=getattr(paths, "eegpt_import", ""),
            model_kwargs_json=getattr(paths, "eegpt_model_kwargs", ""),
            backbone_out_dim=getattr(paths, "eegpt_backbone_out_dim", 512),
            device=device,
            adapter_checkpoint_dir=getattr(paths, "eeg_encoder_path", ""),
            adapter_dim=getattr(paths, "eeg_feature_dim", 512),
            num_classes=getattr(paths, "num_classes", 40),
            input_channels=getattr(paths, "eegpt_input_channels", 128),
            eegpt_channels=getattr(paths, "eegpt_channels", 58),
            target_time_len=getattr(paths, "eegpt_target_time_len", 1024),
        )
    raise ValueError(f"Unknown eeg_encoder_type: {encoder_type}")


def load_llm_model(model_path: str, device: torch.device):
    """Load the frozen LLM from local files.

    The local Mistral config may contain an 8-bit quantization config. That is
    useful on GPU, but it crashes on CPU-only checks, so it is disabled only
    when CUDA is unavailable.
    """

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    load_kwargs: Dict[str, Any] = {
        "config": config,
        "local_files_only": True,
    }

    quantization_dict = getattr(config, "quantization_config", None)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        if isinstance(quantization_dict, dict) and quantization_dict.get("load_in_8bit", False):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    else:
        original_to_dict = config.to_dict

        def to_dict_without_empty_quantization(self):
            quantization_config = getattr(self, "quantization_config", None)
            if quantization_config is None and hasattr(self, "quantization_config"):
                delattr(self, "quantization_config")
            config_dict = original_to_dict()
            if quantization_config is None:
                config_dict.pop("quantization_config", None)
            return config_dict

        config.to_dict = types.MethodType(to_dict_without_empty_quantization, config)
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False
    if not torch.cuda.is_available():
        model = model.to(device)
    return model.eval()


def get_token_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return token embeddings for common causal LLM architectures."""

    input_ids = input_ids.long()
    try:
        return model.get_input_embeddings()(input_ids)
    except Exception:
        pass

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens(input_ids)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte(input_ids)

    raise AttributeError("Cannot find an input embedding layer for this LLM.")


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
