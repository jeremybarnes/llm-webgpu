import torch

if torch.has_mps:
    import mps_fixups
    mps_fixups.fixup_mps()

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/santacoder"
device = "cpu"
#device = "mps"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

print(model)

inputs = tokenizer.encode("def print_first_100_primes():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=200)
print(tokenizer.decode(outputs[0]))