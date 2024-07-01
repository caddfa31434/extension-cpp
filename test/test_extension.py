import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import random
import pytest

from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar,
    Union,
)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


def ref_custom_copy_kv_caches(
    key_caches: List[torch.Tensor],  # List of [num_blocks, block_size, num_heads, head_size]
    value_caches: List[torch.Tensor], # List of [num_blocks, block_size, num_heads, head_size]
    src_slot_mapping: torch.Tensor, # [num_tokens]
    dst_slot_mapping: torch.Tensor,  # [num_tokens]
):
    block_size = value_caches[0].shape[-1]
    assert src_slot_mapping.numel() == dst_slot_mapping.numel()
    for i in range(src_slot_mapping.numel()):
        src_block_number = int(src_slot_mapping[i] // block_size)
        src_block_offset = src_slot_mapping[i] % block_size
        dst_block_number = int(dst_slot_mapping[i] // block_size)
        dst_block_offset = dst_slot_mapping[i] % block_size
        for layer_id in range(len(key_caches)):
            key_cache = key_caches[layer_id]
            value_cache = value_caches[layer_id]

            key_cache[dst_block_number, :, :, dst_block_offset, :] = key_cache[
                src_block_number, :, :, src_block_offset, :
            ]
            value_cache[dst_block_number, :, :, dst_block_offset] = value_cache[
                src_block_number, :, :, src_block_offset
            ]

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 192, 256]
BLOCK_SIZES = [8, 16, 32]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# We assume fp8 is always enabled for testing.
KV_CACHE_DTYPE = ["auto"]

# @pytest.mark.parametrize("num_tokens", NUM_TOKENS)
# @pytest.mark.parametrize("num_layers", NUM_LAYERS)
# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
# @torch.inference_mode()
def test_copy_blocks(
    # kv_cache_factory,
    num_tokens: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    kv_cache_dtype: str,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)


    # Create a random slot mapping.
    num_slots = block_size * num_blocks

    def generate_sorted_random_sequence(num_slots, num_tokens):
        if num_tokens > num_slots:
            raise ValueError("num_tokens cannot be greater than num_slots")
        
        random_sequence = random.sample(range(num_slots), num_tokens)
        sorted_sequence = sorted(random_sequence)
        
        return sorted_sequence


    src_slot_mapping = generate_sorted_random_sequence(num_slots, num_tokens)
    src_slot_mapping = torch.tensor(src_slot_mapping, dtype=torch.long)
    dst_slot_mapping = generate_sorted_random_sequence(num_slots, num_tokens)
    dst_slot_mapping = torch.tensor(dst_slot_mapping, dtype=torch.long)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        num_layers,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    import time

    # Start time
    start_time = time.time()

    # Code block whose execution time you want to measure
    for i in range(1000):
        # Call the move_accepted_kv_cache kernel.
        # key_caches = [kv_cache[0] for kv_cache in kv_caches]
        # value_caches = [kv_cache[1] for kv_cache in kv_caches]
        extension_cpp.ops.custom_copy_kv_caches(key_caches, value_caches, src_slot_mapping, dst_slot_mapping)


    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")


    # Start time
    start_time = time.time()

    # Code block whose execution time you want to measure
    for i in range(1000):
        # Run the reference implementation.
        ref_custom_copy_kv_caches(
            cloned_key_caches, cloned_value_caches, src_slot_mapping, dst_slot_mapping
        )

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches, cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)
    # print(f"{key_caches[0]=}")
    # print(f"{cloned_key_caches[0]=}")

if __name__ == "__main__":
    test_copy_blocks(num_tokens=5, num_layers=32, num_heads=32, head_size=128, block_size=16, num_blocks=1000, dtype=torch.float16, seed=0, kv_cache_dtype="auto", device="cuda")