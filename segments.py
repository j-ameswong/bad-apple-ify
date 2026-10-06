"""Compatibility facade for resumable segmented encoding."""

from bad_apple.segments import (CHECKPOINT_VERSION, CODE_CHECKPOINT_VERSION,
                                MANIFEST_NAME, encode_segmented)

__all__ = ["CHECKPOINT_VERSION", "CODE_CHECKPOINT_VERSION", "MANIFEST_NAME",
           "encode_segmented"]
