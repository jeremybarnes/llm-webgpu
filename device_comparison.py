import torch
from torch.nn import Module
from typing import List, Tuple, Dict, Generator, Any, Optional
from tensor_comparisons import TensorDifference
import copy

def fixup_argument(arg: Any, device: torch.device, dtype: Optional[torch.dtype] = None):
    #print('fixup argument', type(arg), isinstance(arg, Generator))
    if isinstance(arg, torch.Tensor):
        if arg.storage_offset() > 0 and dtype is None:
            #print("storage_offset =", arg.storage_offset())
            old_storage = arg.storage()
            new_storage = torch.TypedStorage(old_storage.size(), device=device, dtype=old_storage.dtype).copy_(old_storage)
            res = torch.Tensor().to(device, arg.dtype)
            res.set_(new_storage, arg.storage_offset(), arg.size(), arg.stride())
            #print("storage_offset =", arg.storage_offset(), "new storage offset =", res.storage_offset())
            assert res.storage_offset() == arg.storage_offset()
            assert res.size() == arg.size()
            assert res.stride() == arg.stride()
        else:
            # dtype is only changed for floating point types, otherwise we just shift device
            if dtype is not None and arg.dtype.is_floating_point:
                res = arg.to(device, dtype)
            else:
                res = arg.to(device)
        return res
    elif isinstance(arg, List) or isinstance(arg, Generator):
        return list([fixup_argument(a, device, dtype) for a in arg])
    elif isinstance(arg, Tuple):
        return tuple((fixup_argument(a, device, dtype) for a in arg))
    elif isinstance(arg, Dict):
        return dict({k: fixup_argument(v, device, dtype) for k,v in arg.items()})
    else:
        return arg

def compare_output(expected: Any, received: Any, name: str) -> TensorDifference:
    #print("compare: expected ", expected, "received", received)
    if type(expected) != type(received):
        # Coerce to tuple if one is a tuple
        if isinstance(expected, tuple) or isinstance(received, tuple):
            return compare_output(tuple(expected), tuple(received), name)
        # Coerce to dict if one is a dict
        if isinstance(expected, dict) or isinstance(received, dict):
            return compare_output(dict(expected), dict(received), name)
        raise RuntimeError(f'{name} types differ: {type(expected)} vs {type(received)}')

    if isinstance(expected, torch.Tensor):
        if expected.shape != received.shape:
            raise RuntimeError(f'{name} tensor shapes differ: {expected.shape} vs {received.shape}')
        return TensorDifference.between(expected, received.to(expected.device))

    elif isinstance(expected, list) or isinstance(expected, tuple):
        if len(expected) != len(received):
            raise RuntimeError("{name} lengths differ: {len(expected)} vs {len(received})")
        res = TensorDifference()
        for i in range(len(expected)):
            res2 = compare_output(expected[i], received[i], name + "[" + str(i) + "]")
            if res.ulps < res2.ulps:
                res = res2
        return res
    elif isinstance(expected, dict):
        items1 = sorted(expected.items())
        items2 = sorted(received.items())

        keys1 = [k for k,_ in items1]
        keys2 = [k for k,_ in items2]

        if keys1 != keys2:
            raise RuntimeError(f"{name} dict keys differ: {keys1} vs {keys2}")

        # Compare the values for each key
        res = TensorDifference()
        for i in range(len(keys1)):
            k,v1 = items1[i]
            _,v2 = items2[i]
            res2 = compare_output(v1, v2, name + "{" + k + "}")
            if res.ulps < res2.ulps:
                res = res2
        return res
    else:
        if expected == received:
            return TensorDifference()
        raise RuntimeError(f'{name} outputs differ: {expected} vs {received}')

def device_comparison_mode(model: Module, master_device: torch.device, master_dtype: torch.dtype, scenarios: List[Tuple[torch.device, torch.dtype]]):
    """
    Instrument the given model to run on multiple devices/datatypes, comparing the results
    between the (canonical) CPU version and the others, to identify where accuracy errors
    are coming from.
    """

    def recurse_module(m: Module, recursion: int, path: str):

        n: str = m._get_name()
        indent: str = ' ' * recursion * 4

        slave_modules: List[Tuple[Module,torch.device,torch.dtype]] = []

        for device,dtype in scenarios:
            slave_modules.append((copy.deepcopy(m).to(torch.device(device), dtype), device, dtype))

        def pre_hook(module: Module, args: Tuple, kwargs: Dict):
            #print(f"    {path:60} {n} ENTER")
            return None

        def post_hook(module: Module, args: Tuple, kwargs: Dict, output: Tuple):
            master_output = output
            #master_output = module.forward(*args, **kwargs)

            debug: bool = n == "Embedding"
            if debug:
                #args[0].contiguous()
                #while True:
                #    master_output = module.forward(*args, **kwargs)

                print(f"Embedding: input {args[0].shape}", args[0])
                print(f"Embedding: output {master_output[0].shape}", master_output[0][0][0:10])
                assert isinstance(module, torch.nn.Embedding)
                print("weights are", module.weight.shape)
                idx = args[0][0][0].item()
                print("index weights are", module.weight[idx][0:10])
                index: Tensor = args[0]
                print(f"index.storage = { index.storage() } offset = { index.storage_offset() })")
                if module.weight[idx].ne(master_output[0][0]).any():
                    for i in range(module.weight.shape[0]):
                        if module.weight[i].eq(master_output[0][0]).all():
                            print('equal i = ', i)

                    raise RuntimeError("embedding got the wrong thing")

            # jit._get_trace_graph
            #graph = jit.trace(m.forward, example_inputs=args, example_kwarg_inputs=kwargs)

            for slave,device,dtype in slave_modules:
                prefix = f"    {path:50} {n:20} {device}"

                if debug:
                    assert isinstance(slave, torch.nn.Embedding)
                    assert isinstance(module, torch.nn.Embedding)
                    assert module.weight.to('cpu').eq(slave.weight.to('cpu')).all()


                try:
                    def fixup_to_slave(arg):
                        return fixup_argument(arg, device, dtype)

                    def fixup_to_master(arg):
                        return fixup_argument(arg, master_device, master_dtype)

                    if False:
                        for a in args:
                            print("arg ", type(a))
                        for k,v in kwargs.items():
                            print("kw", k, type(v))
                        if 'layer_past' in kwargs:
                            print('layer_past', kwargs['layer_past'])

                    slave_output = slave.forward(*fixup_to_slave(args), **{k: fixup_to_slave(v) for k,v in kwargs.items()})

                    if debug:
                        print(f"Embedding: slave output {slave_output[0].shape}", slave_output[0][0][0:10])
                        print("index weights are", slave.weight[idx][0:10])

                    slave_output = fixup_to_master(slave_output)

                except Exception as e:
                    print(prefix + f" FAILED {e}")
                    raise
                    return None

                biggest: List[Optional[TensorDifference]] = [None]

                compare_output(master_output, slave_output, str(device) + " " + str(dtype) + " " + path)
                print(prefix + f" {biggest[0].ulps:>5} ulps")
                if biggest[0].ulps > 1000:
                    raise RuntimeError("too big an ULPS difference")

            return None

        m.register_forward_pre_hook(pre_hook, with_kwargs=True)
        m.register_forward_hook(post_hook, with_kwargs = True)

        for name,child in m.named_children():
            recurse_module(child, recursion + 1, path + "." + name)

    recurse_module(model, 0, '')
