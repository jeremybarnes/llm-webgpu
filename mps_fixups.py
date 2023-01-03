from torch.library import Library as _Library
import torch
_lib = _Library("aten", "IMPL")

def fixup_mps():

    # Topk only works up to k=15 on MPS, replace it with a CPU fallback
    def _topk(self: torch.Tensor, k: int, dim:int=-1, largest:bool=True, sorted:bool=True):
        res, indices = torch.topk(self.to('cpu', torch.float32), k, dim, largest, sorted)
        return res.to(self), indices.to('mps')

    _lib.impl("topk", _topk, "MPS")

    # Max doesn't work with longs on MPS, replace it with a CPU fallback
    def _max(self: torch.Tensor) -> torch.Tensor:
        return torch.max(self.to('cpu')).to('mps')

    _lib.impl("max", _max, "MPS")

    # embedding desn't work with sliced tensors (in fact all index operations are broken when storage_offset() != 0).
    # eventually we will fix here, currently the fix is in PyTorch
    #def _embedding(weight: torch.Tensor, input: torch.Tensor, padding_idx: int = -1, scale_grad_by_freq: bool = False, sparse: bool = False):
    #    return torch.embedding(weight.to('cpu'), input.to('cpu'), padding_idx, scale_grad_by_freq, sparse).to('mps')

    #_lib.impl("embedding", _embedding, "MPS")

