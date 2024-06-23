import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

if __name__ == "__main__":
    a = torch.randn(3, device="cuda")
    b = torch.randn(3, device="cuda")
    out = torch.zeros(3, device="cuda")
    print(f"{a=}")
    print(f"{b=}")
    print(f"{out=}")
    extension_cpp.ops.myadd_out(a, b, out)
    print(f"{a=}")
    print(f"{b=}")
    print(f"{out=}")