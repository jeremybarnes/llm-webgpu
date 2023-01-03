import torch
from torch.nn import Module
from typing import List, Tuple, Dict
from tensor_comparisons import TensorDifference
import time

def instrument_runtimes(model: Module, output: Dict[str, float]):
    """
    Instrument the given model to collect runtimes of each type of layer in the given dict
    """

    def recurse_module(m: Module, recursion: int, path: str):

        n: str = m._get_name()
        indent: str = ' ' * recursion * 4
        start: List[float] = [0]

        def pre_hook(module: Module, args: Tuple, kwargs: Dict):
            #print(f"{indent}{n} ENTER")
            start[0] = time.time()
            return None

        def post_hook(module: Module, args: Tuple, kwargs: Dict, output: Tuple):
            finish = time.time()
            elapsed = finish - start[0]
            if elapsed > 0.001:
                s = path + " " + n
                print(f"    {s:60} {int(elapsed*100000)*10:10}us")
            return None

        m.register_forward_pre_hook(pre_hook, with_kwargs=True)
        m.register_forward_hook(post_hook, with_kwargs = True)

        for name,child in m.named_children():
            recurse_module(child, recursion + 1, path + "." + name)

    recurse_module(model, 0, '')
