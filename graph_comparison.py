from graphs import Scope, Operation, default_find_operation
from torch._C import Graph, Node, dtype as cdtype
from device_comparison import fixup_argument, compare_output
from tensor_comparisons import TensorDifference
from runtimes import instrument_runtimes, print_elapsed
import torch
from typing import Callable, Any
import time


def find_operation(n: Node) -> Operation:
    return default_find_operation(n)

def make_time_operation(n: Node) -> Operation:
    old_op = default_find_operation(n)
    new_op = Operation(old_op.name + '.timed', old_op.source)

    def new_interpreter(s: Scope, n: Node) -> Any:
        before = time.time()
        res = old_op.interpret(s, n)
        after = time.time()
        print(n.kind(), print_elapsed(after-before))
        return res

    new_op._interpreter = new_interpreter
    return new_op

def make_compare_operation(device: torch.device|str, time_only: bool = False) -> Callable[[Node],Operation]:

    def to_device(val: Any) -> Any:
        return fixup_argument(val, torch.device(device))

    def update_scope(s1: Scope, s2: Scope):
        for i in range(len(s2.vars), len(s1.vars)):
            name,val=s1.vars[i]
            s2.add_var(name, to_device(val))

    def compare_operation(n: Node) -> Operation:
        old_op = default_find_operation(n)
        new_op = Operation(old_op.name + '.compare', old_op.source)

        scopes: Dict[Scope, Scope] = {}

        def new_interpreter(s1: Scope, n: Node) -> Any:
            if s1 not in scopes:
                scopes[s1] = Scope()

            s2: Scope = scopes[s1]
            update_scope(s1, s2)
            len_before = len(s1.vars)

            before1 = time.time()
            res1 = old_op.interpret(s1, n)
            after1 = time.time()

            before2 = time.time()
            res2 = old_op.interpret(s2, n)
            after2 = time.time()

            if time_only:
                print(f"{n.kind():30} {print_elapsed(after1-before1):8} {print_elapsed(after2-before2):8}")
                return res1

            max_difference = TensorDifference()
            for i in range(len_before, len(s1.vars)):
                name1,val1 = s1.vars[i]
                name2,val2 = s2.vars[i]

                assert name1 == name2

                try:
                    difference = compare_output(val1, val2, name1)
                    if max_difference.ulps < difference.ulps:
                        max_difference = difference
                except Exception as e:
                    if str(e).find('cpu') == -1:
                        print(f"{fg.red}{e}{reset}")
                        raise

                #s2.vars[i] = (name2, to_device(val1))

            print(f"{n.kind():30} {print_elapsed(after1-before1):8} {print_elapsed(after2-before2):8} {max_difference.ulps:6} ulps")
            return res1

        new_op._interpreter = new_interpreter
        return new_op

    return compare_operation
