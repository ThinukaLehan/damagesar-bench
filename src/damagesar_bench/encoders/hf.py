"""Hugging Face ViT encoders (needs the ``ml`` extra and, for DINOv3, ``HF_TOKEN``).

The single SAR channel is copied to 3 channels, then normalised with the model's own mean and
SD. (Copying the channel is mathematically the same as summing the first-layer weights for a
1-channel input; the open adapter question in the pre-registration is only about the
normalisation, decided on development events.) Prefix tokens (CLS and any register tokens)
are dropped, so the output is the patch-token grid only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import Encoder, EncoderError, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tokens_to_grid(hidden: np.ndarray, h: int, w: int) -> np.ndarray:
    """(1, n_prefix + h*w, dim) model output -> (h, w, dim) patch grid.

    The prefix (CLS + register tokens) is whatever comes before the last h*w tokens; it must be
    1 to 5 tokens (ViT/DINOv2: 1; DINOv3: 1 CLS + 4 registers).
    """
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise EncoderError(f"expected (1, tokens, dim), got {hidden.shape}")
    n_prefix = hidden.shape[1] - h * w
    if not 1 <= n_prefix <= 5:
        raise EncoderError(f"{hidden.shape[1]} tokens for a {h}x{w} grid: unexpected prefix of {n_prefix}")
    return hidden[0, n_prefix:].reshape(h, w, hidden.shape[2])


def resolve_weights_revision(model_id: str, requested: str | None, config=None, cache_dir=None) -> str | None:
    """The Hugging Face commit the weights were loaded from.

    Newer transformers versions no longer set ``config._commit_hash``, so fall back to the local cache:
    the cached files live under ``snapshots/<commit>/``. No network call. Returns the requested
    revision (or None) only if neither source has it.
    """
    commit = getattr(config, "_commit_hash", None) if config is not None else None
    if commit:
        return commit
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, "config.json", revision=requested, cache_dir=cache_dir)
    except Exception:  # pragma: no cover - depends on the environment
        path = None
    if isinstance(path, str):
        parts = Path(path).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return requested


class HFViT(Encoder):
    model_id = ""
    revision: str | None = None   # Hugging Face commit to load; None = latest (the run records which it got)
    needs_interpolation = False   # True for models with fixed 224 px position embeddings
    norm_stats: tuple | None = None  # (mean, std) per RGB channel; None = ImageNet statistics

    def __init__(self, device: str | None = None, fp16: bool = True):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as ex:  # pragma: no cover - depends on the environment
            raise EncoderError("install the ml extra: pip install -e '.[ml]'") from ex
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        rev = os.environ.get(f"DSB_REVISION_{self.name.upper()}") or self.revision
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype, revision=rev).to(self.device).eval()
        # The exact weights used: the Hugging Face commit the files came from (recorded for reproducibility).
        self.weights_revision = resolve_weights_revision(self.model_id, rev, self.model.config)
        self.dim = int(self.model.config.hidden_size)
        self.patch_px = int(self.model.config.patch_size)
        mean, std = self.norm_stats or (IMAGENET_MEAN, IMAGENET_STD)
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def _encode(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch + weights
        torch = self._torch
        x = (np.repeat(image[None], 3, axis=0) - self.mean) / self.std
        t = torch.from_numpy(x[None]).to(self.device, self.dtype)
        kwargs = {"interpolate_pos_encoding": True} if self.needs_interpolation else {}
        with torch.inference_mode():
            hidden = self.model(pixel_values=t, **kwargs).last_hidden_state.float().cpu().numpy()
        h, w = image.shape[0] // self.patch_px, image.shape[1] // self.patch_px
        return tokens_to_grid(hidden, h, w)


@register("dinov3_vitb16")
class DINOv3ViTB16(HFViT):
    name = "dinov3_vitb16"
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@register("dinov3_vitl16")
class DINOv3ViTL16(HFViT):
    """DINOv3 ViT-L/16, web images (LVD-1689M): the size-matched partner of the satellite model."""
    name = "dinov3_vitl16"
    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"


@register("dinov3_vitl16_sat")
class DINOv3ViTL16Sat(HFViT):
    """DINOv3 ViT-L/16 pretrained on satellite images (SAT-493M), with the statistics Meta gives for it."""
    name = "dinov3_vitl16_sat"
    model_id = "facebook/dinov3-vitl16-pretrain-sat493m"
    norm_stats = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))


@register("dinov2_vitb14")
class DINOv2ViTB14(HFViT):
    name = "dinov2_vitb14"
    model_id = "facebook/dinov2-base"


@register("imagenet_vitb16")
class ImageNetViTB16(HFViT):
    name = "imagenet_vitb16"
    model_id = "google/vit-base-patch16-224"
    needs_interpolation = True
