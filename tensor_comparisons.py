# Tensor comparisons that can identify significant differences taking into account
# different devices, precisions and implementations of the calculations.

import torch
import struct
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

def float16_to_int(i: Tensor) -> Tensor:
    assert i.storage_offset() == 0
    #print(f"stride {i.stride()} storage offset {i.storage_offset()}")
    if i.dtype != torch.float16:
        raise RuntimeError(f"expected fp16 tensor; got {i.dtype}")
    #print(f"shape in {i.shape} shape out {torch.ShortTensor(i.to('cpu').untyped_storage()).shape}")
    return torch.ShortTensor(i.to('cpu').untyped_storage()).reshape_as(i)

def int_to_float16(i: Tensor) -> Tensor:
    assert i.storage_offset() == 0
    if not isinstance(i, torch.ShortTensor):
        raise RuntimeError(f"expected i16 tensor; got {type(i)}")
    return torch.HalfTensor(i.to('cpu').untyped_storage()).reshape_as(i)

def bfloat16_to_int(i: torch.Tensor) -> torch.ShortTensor:
    if not isinstance(i.storage(), torch.BFloat16Storage):
        raise RuntimeError(f"expected bf16 tensor; got {type(i)}")
    return torch.ShortTensor(i.untyped_storage())

def int_to_bfloat16(i: torch.ShortTensor) -> Tensor:
    if not isinstance(i, torch.ShortTensor):
        raise RuntimeError(f"expected i16 tensor; got {type(i)}")
    return torch.Tensor(torch.BFloat16Storage(i.untyped_storage()))

def float32_to_int(i: torch.FloatTensor) -> torch.IntTensor:
    return torch.IntTensor(i.untyped_storage())

def int_to_float32(i: torch.IntTensor) -> torch.FloatTensor:
    return torch.FloatTensor(i.untyped_storage())

def float64_to_int(i: torch.DoubleTensor) -> torch.LongTensor:
    return torch.LongTensor(i.untyped_storage())

def int_to_float64(i: torch.IntTensor) -> torch.DoubleTensor:
    return torch.DoubleTensor(i.untyped_storage())

# Maximum number of floating point ULPS that we accept as a difference in a computation
MAX_ULPS=300

# Maximum absolute difference we accept as a difference in a computation
MAX_DIFF=1e-6



def ulps_difference(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    """
    Returns the number of ulps in difference between the two tensors, taking into account overflow
    """

    return torch.minimum(torch.abs(i1 - i2), 32768 - torch.abs(i2 - i1))

    diff1 = i1 - i2
    diff2 = i2 - i1

    return torch.minimum(diff1, diff2)

@dataclass
class TensorDifference:
    expected: Tensor
    received: Tensor
    
    index: int = 0
    v1: Optional[Tensor] = None
    v2: Optional[Tensor] = None

    i1: int = 0
    i2: int = 0

    difference: float = 0.
    ulps: int = 0

    message: str = ''

    def __bool__(self) -> bool:
        return self.message != '' or self.difference != 0

    def is_significant(self) -> bool:
        """
        Returns true if and only if the difference between the values is considered
        significant.
        """

        return abs(self.i1 - self.i2) > MAX_ULPS

    @staticmethod
    def between(expected: Tensor, received: Tensor) -> 'TensorDifference':
        """
        Compare the two tensors, first converting to a common representation to ensure that different
        data types are taken into account.

        Returns None if there is no difference, or a string describing it if there is one.
        """

        result = TensorDifference(expected, received)

        sh1,dt1 = expected.shape, expected.dtype
        sh2,dt2 = received.shape, received.dtype

        if sh1 != sh2:
            result.message = "tensor shapes differ: {sh1} vs {sh2}"
            return result

        if expected.eq(received).all():
            return result

        flat1 = expected.to('cpu').flatten()
        flat2 = received.to('cpu').flatten()

        if not dt1.is_floating_point:
            diff = torch.abs(flat2 - flat1)
            result.difference = diff.max().item()
            result.index = int(diff.argmax().item())

        else:
            if False:
                # floating point

                tinfo = {
                    torch.float16:   (2, torch.int16,  float16_to_int, int_to_float16),
                    torch.bfloat16:  (2, torch.int16,  bfloat16_to_int, int_to_bfloat16),
                    torch.float32:   (4, torch.int32,  float32_to_int, int_to_float32),
                    torch.float64:   (8, torch.int64,  float64_to_int, int_to_float64),
                }

                w1,it1,to_int1,from_int1 = tinfo[dt1]
                w2,it2,to_int2,from_int2 = tinfo[dt2]

                if w2 < w1:
                    dt,w,it,to_int,from_int = dt2,w2,it2,to_int2,from_int2
                else:
                    dt,w,it,to_int,from_int = dt1,w1,it1,to_int1,from_int1

            # Temporary
            dt = torch.float16
            to_int = float16_to_int
            from_int = int_to_float16

            #print(f'dt = {dt} dt1 = {dt1} dt2 = {dt2}')

            conv1 = flat1.to(dt)
            conv2 = flat2.to(dt)

            if conv1.eq(conv2).all():
                return result

            asint1 = to_int(conv1)
            asint2 = to_int(conv2)

            assert from_int(asint1).eq(conv1).all()
            assert from_int(asint2).eq(conv2).all()

            diff = ulps_difference(asint1, asint2)

            max_diff = diff.max()
            max_diff_el = torch.argmax(diff)
            result.ulps = int(max_diff.item())
            result.index = int(max_diff_el.item())
            result.i1 = int(asint1[result.index])
            result.i2 = int(asint2[result.index])
            result.difference = float(conv2[max_diff_el] - conv1[max_diff_el].item())

            assert torch.flatten(diff)[max_diff_el] == max_diff

            #el1 = torch.flatten(expected)[max_diff_el]
            #el2 = torch.flatten(received)[max_diff_el]
            #eli1 = torch.flatten(asint1)[max_diff_el]
            #eli2 = torch.flatten(asint2)[max_diff_el]

            #print(f'max_diff_fp = {max_diff_fp} max_diff = {max_diff} max_diff_el = {max_diff_el} dt1 = {dt1} dt2 = {dt2} dt = {dt} shape = {sh1} el1 = {el1} el2 = {el2} eli1 = {eli1} eli2 = {eli2}')
            #
            #if max_diff <= MAX_ULPS or max_diff >= 32768 - MAX_ULPS:
            #    return None

            #return f"tensors differ; max diff is {max_diff} ulps"

        result.v1 = flat1[result.index]
        result.v2 = flat2[result.index]
        return result
