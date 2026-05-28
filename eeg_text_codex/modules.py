import importlib
import json
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection


class MultiHead(nn.Module):
    """Map a single EEG embedding into low / mid / high semantic branches.

    This class mirrors the first-stage head architecture so existing
    `best_multi_head.pth` checkpoints can be loaded directly.
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.head_low = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.head_mid = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))
        self.head_high = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, out_dim))

    def forward(self, x: torch.Tensor):
        return self.head_low(x), self.head_mid(x), self.head_high(x)


class CandidateReranker(nn.Module):
    """Score candidate object labels from retrieval/classifier evidence."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class EnhancedSemanticMOE(nn.Module):
    """Adaptive fusion over low / mid / high EEG semantic features.

    Each semantic branch first goes through its own expert MLP. A gate then
    predicts per-sample weights, so the model can emphasize different semantic
    levels for different EEG examples.
    """

    def __init__(self, feat_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.expert_low = self._make_expert(feat_dim, hidden_dim, out_dim, dropout)
        self.expert_mid = self._make_expert(feat_dim, hidden_dim, out_dim, dropout)
        self.expert_high = self._make_expert(feat_dim, hidden_dim, out_dim, dropout)

        self.gate = nn.Sequential(
            nn.LayerNorm(feat_dim * 3),
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    @staticmethod
    def _make_expert(feat_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, feat_low: torch.Tensor, feat_mid: torch.Tensor, feat_high: torch.Tensor):
        e_low = self.expert_low(feat_low)
        e_mid = self.expert_mid(feat_mid)
        e_high = self.expert_high(feat_high)

        gate_logits = self.gate(torch.cat([feat_low, feat_mid, feat_high], dim=-1))
        gate_weight = F.softmax(gate_logits, dim=-1)
        fused = (
            gate_weight[:, 0:1] * e_low
            + gate_weight[:, 1:2] * e_mid
            + gate_weight[:, 2:3] * e_high
        )
        return self.out_norm(fused), gate_weight


class ProjectionLayer(nn.Module):
    """Project fused EEG semantics to the frozen LLM hidden size."""

    def __init__(self, in_dim: int = 512, out_dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EEGPTAdapterClassifier(nn.Module):
    """Frozen EEGPT backbone plus trainable adapter and classifier.

    The wrapper intentionally exposes the same forward contract as ChannelNet:
    it returns `(eeg_feature, class_logits)`, where `eeg_feature` is 512d by
    default. This lets the later semantic heads and LLM stages stay simple.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        adapter_dim: int = 512,
        num_classes: int = 40,
        dropout: float = 0.1,
        input_channels: int = 128,
        eegpt_channels: int = 58,
        target_time_len: int = 1024,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_out_dim = backbone_out_dim
        self.adapter_dim = adapter_dim
        self.num_classes = num_classes
        self.target_time_len = target_time_len
        self.channel_adapter = nn.Conv1d(input_channels, eegpt_channels, kernel_size=1)
        self.adapter = nn.Sequential(
            nn.LayerNorm(backbone_out_dim),
            nn.Linear(backbone_out_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, num_classes),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @property
    def trainable_parameters(self):
        return (
            list(self.channel_adapter.parameters())
            + list(self.adapter.parameters())
            + list(self.classifier.parameters())
        )

    @staticmethod
    def _pool_backbone_output(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            for key in ("last_hidden_state", "features", "feature", "x"):
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(value for value in output.values() if torch.is_tensor(value))
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if output.dim() == 2:
            return output
        if output.dim() == 3:
            return output.mean(dim=1)
        if output.dim() == 4:
            return output.mean(dim=1).flatten(start_dim=1)
        return output.flatten(start_dim=1)

    def extract_backbone_feature(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.dim() == 4 and eeg.size(1) == 1:
            eeg = eeg.squeeze(1)
        eeg = eeg.float()
        eeg = eeg - eeg.mean(dim=-2, keepdim=True)
        if self.target_time_len > 0 and eeg.shape[-1] != self.target_time_len:
            eeg = F.interpolate(eeg, size=self.target_time_len, mode="linear", align_corners=False)
        eeg = self.channel_adapter(eeg)
        output = self.backbone(eeg)
        return self._pool_backbone_output(output)

    def forward(self, eeg: torch.Tensor):
        backbone_feat = self.extract_backbone_feature(eeg)
        eeg_feat = self.adapter(backbone_feat.float())
        logits = self.classifier(eeg_feat)
        return eeg_feat, logits

    def save_adapter(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "classifier": self.classifier.state_dict(),
                "channel_adapter": self.channel_adapter.state_dict(),
                "backbone_out_dim": self.backbone_out_dim,
                "adapter_dim": self.adapter_dim,
                "num_classes": self.num_classes,
                "metadata": metadata or {},
            },
            os.path.join(output_dir, "encoder.pt"),
        )

    def load_adapter(self, checkpoint_dir: str, device: torch.device) -> None:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "encoder.pt"), map_location=device)
        if "channel_adapter" in checkpoint:
            self.channel_adapter.load_state_dict(checkpoint["channel_adapter"])
        self.adapter.load_state_dict(checkpoint["adapter"])
        self.classifier.load_state_dict(checkpoint["classifier"])


def _import_from_spec(import_spec: str):
    module_name, class_name = import_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_eegpt_backbone(
    model_dir: str,
    import_spec: str,
    checkpoint_path: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Build an EEGPT backbone from local source code and local checkpoint.

    `import_spec` should look like `archs.EEGPT_mcae:EEGTransformer`.
    `model_dir` is appended to `sys.path` before importing. The checkpoint can
    be a raw state dict, a dict containing `state_dict` / `model`, or a whole
    serialized module.
    """

    if model_dir and model_dir not in sys.path:
        sys.path.append(model_dir)
    model_kwargs = model_kwargs or {}
    model_cls = _import_from_spec(import_spec)
    backbone = model_cls(**model_kwargs)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, nn.Module):
            backbone = checkpoint
        else:
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            cleaned = {}
            for key, value in state_dict.items():
                for prefix in ("module.", "backbone.", "student.", "encoder.", "target_encoder."):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                cleaned[key] = value
            backbone.load_state_dict(cleaned, strict=False)
    return backbone.eval()


def load_eegpt_model_kwargs(model_kwargs_json: str) -> Dict[str, Any]:
    if not model_kwargs_json:
        return {}
    if os.path.exists(model_kwargs_json):
        with open(model_kwargs_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(model_kwargs_json)


class CLIPSemanticTargetEncoder(nn.Module):
    """Extract low / mid / high image semantic targets from one frozen CLIP ViT.

    Low and mid targets are mean-pooled patch tokens from intermediate ViT
    hidden states, projected through CLIP's visual projection so all targets
    live in the same 512-dimensional CLIP embedding space as the final image
    embedding.
    """

    def __init__(
        self,
        clip_path: str,
        low_layers=(3, 4),
        mid_layers=(7, 8),
    ):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_path, local_files_only=True)
        self.low_layers = tuple(low_layers)
        self.mid_layers = tuple(mid_layers)
        for parameter in self.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _pool_patch_tokens(hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state[:, 1:, :].mean(dim=1)

    def _project_hidden(self, hidden_states, layers) -> torch.Tensor:
        pooled = torch.stack([self._pool_patch_tokens(hidden_states[layer]) for layer in layers], dim=0).mean(dim=0)
        return self.clip.visual_projection(pooled)

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.clip(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        low = self._project_hidden(hidden_states, self.low_layers)
        mid = self._project_hidden(hidden_states, self.mid_layers)
        high = outputs.image_embeds
        return low, mid, high
