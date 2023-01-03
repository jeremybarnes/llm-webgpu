import torch
from torch.library import Library as _Library
_lib = _Library("aten", "IMPL")

# Topk only works up to k=15 on MPS, replace it with a CPU fallback
def _topk(self: torch.Tensor, k: int, dim:int=-1, largest:bool=True, sorted:bool=True):
    res, indices = torch.topk(self.to('cpu'), k, dim, largest, sorted)
    return res.to('mps'), indices.to('mps')

_lib.impl("topk", _topk, "MPS")

# Max doesn't work with longs on MPS, replace it with a CPU fallback
def _max(self: torch.Tensor) -> torch.Tensor:
    return torch.max(self.to('cpu')).to('mps')

_lib.impl("max", _max, "MPS")


from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/santacoder"
device = "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

print(model)

inputs = tokenizer.encode("def print_first_100_primes():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=200)
print(tokenizer.decode(outputs[0]))