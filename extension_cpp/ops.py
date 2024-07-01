import torch
from torch import Tensor
from typing import List

__all__ = ["myadd_out", "custom_copy_kv_caches"]


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)


def custom_copy_kv_caches(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    src_slot_mappings: Tensor,
    dst_slot_mappings: Tensor,
) -> None:
    torch.ops.extension_cpp.custom_copy_kv_caches.default(key_caches, value_caches, src_slot_mappings, dst_slot_mappings)
