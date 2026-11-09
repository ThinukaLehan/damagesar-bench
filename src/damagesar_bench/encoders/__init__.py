"""Encoder plugins (FR5). A new encoder is one class in one file, registered with ``@register``.

Built in: ``patchstats`` (hand-made patch statistics, no learning; tests and simple baseline),
``imagenet_vitb16``, ``dinov2_vitb14``, ``dinov3_vitb16`` (Hugging Face; need the ``ml`` extra).
"""

from .base import Encoder, EncoderError, available, get_encoder, register  # noqa: F401
