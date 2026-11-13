"""Hugging Face ViT encoders (needs the ``ml`` extra and, for DINOv3, ``HF_TOKEN``).

The single SAR channel is copied to 3 channels, then normalised with the model's own mean and
SD. (Copying the channel is mathematically the same as summing the first-layer weights for a
1-channel input; the open adapter question in the pre-registration is only about the
normalisation, decided on development events.) Prefix tokens (CLS and any register tokens)
are dropped, so the output is the patch-token grid only.
"""

from __future__ import annotations

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


class HFViT(Encoder):
    model_id = ""
    needs_interpolation = False   # True for models with fixed 224 px position embeddings

    def __init__(self, device: str | None = None, fp16: bool = True):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as ex:  # pragma: no cover - depends on the environment
            raise EncoderError("install the ml extra: pip install -e '.[ml]'") from ex
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        self.patch_px = int(self.model.config.patch_size)
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        self.std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]

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


@register("dinov2_vitb14")
class DINOv2ViTB14(HFViT):
    name = "dinov2_vitb14"
    model_id = "facebook/dinov2-base"


@register("imagenet_vitb16")
class ImageNetViTB16(HFViT):
    name = "imagenet_vitb16"
    model_id = "google/vit-base-patch16-224"
    needs_interpolation = True
