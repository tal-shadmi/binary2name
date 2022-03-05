import angr
import re

bases_dict = dict()
replacement_dict = dict()
"""
    <Bool reg_rdi_15_64{UNINITIALIZED} == 0xfffffe0000000000>,
    <Bool mem_fffffe0000000000_17_32{UNINITIALIZED} <=s mem_fffffe0000000004_18_32{UNINITIALIZED}>,
    <Bool mem_fffffe0000000004_18_32{UNINITIALIZED} <=s mem_fffffe0000000008_19_32{UNINITIALIZED}>
    <Bool reg_rdi_15_64{UNINITIALIZED} == 0xfffffe0000000000>,
    <Bool mem_fffffe0000000000_17_32{UNINITIALIZED} <=s mem_fffffe0000000004_18_32{UNINITIALIZED}>,
    <Bool !(mem_fffffe0000000004_18_32{UNINITIALIZED} <=s mem_fffffe0000000008_19_32{UNINITIALIZED})>,
    <Bool mem_fffffe0000000000_17_32{UNINITIALIZED} <=s mem_fffffe0000000008_19_32{UNINITIALIZED}>
    
    <Bool reg_rdi_15_64{UNINITIALIZED} == 0xfffffe0000000000>,
    <Bool !(mem_fffffe0000000000_17_32{UNINITIALIZED} <=s mem_fffffe0000000004_18_32{UNINITIALIZED})>,
    <Bool mem_fffffe0000000000_17_32{UNINITIALIZED} <=s mem_fffffe0000000008_20_32{UNINITIALIZED}>,
    <Bool mem_fffffe0000000004_18_32{UNINITIALIZED} <=s mem_fffffe0000000000_17_32{UNINITIALIZED}>
"""


def address_breakfun(state):
    # if not state.inspect.address_concretization_add_constraints:
    #     return
    if state.inspect.address_concretization_result is None:
        return

    # if len(state.inspect.address_concretization_expr.args) == 1:
    #     bases.add()
    # print(state.inspect.address_concretization_expr.op)
    # print(f"{hex(state.inspect.address_concretization_result[0])}")
    # print(f"{state.inspect.address_concretization_expr}")
    expr = state.inspect.address_concretization_expr
    if expr.depth > 2:
        raise Exception("should consider a new approach, your assumption is wrong!!")
    if expr.depth == 1:
        if expr.op != "BVS":
            raise Exception("AAAAAA!!!")
        if state.solver.eval(expr) in bases_dict:
            return
        # new var is declared
        var_name = f"var_{len(bases_dict)}"
        bases_dict[state.inspect.address_concretization_result[0]] = var_name
        replacement_dict[state.inspect.address_concretization_result[0]] = f"{var_name}(0)"
    else:
        # depth is 2 (either a new sym-var is being declared or offset calc)
        if expr.op != "__add__":
            raise Exception("AAAAAA!!!")
        childs = list(expr.args)
        assert len(childs) < 3
        if len(childs) == 1:
            if state.solver.eval(expr) in bases_dict:
                return
            # new var is declared
            var_name = f"var_{len(bases_dict)}"
            bases_dict[state.inspect.address_concretization_result[0]] = var_name
            replacement_dict[state.inspect.address_concretization_result[0]] = f"{var_name}(0)"
        else:
            base = None
            offset = None
            for c in childs:
                if not c.concrete:
                    base = state.solver.eval(c)
                else:
                    offset = state.solver.eval(c)
            replacement_dict[state.inspect.address_concretization_result[0]] = f"{bases_dict[base]}({offset})"


# constraints
def my_bp(state):
    pass


# symbolic_variable
def my_bp2(state):
    pass


def main():
    proj = angr.Project("./core_utils_bins/head", auto_load_libs=False)
    proj.analyses.CFGFast()
    fn = proj.kb.functions['head_bytes']
    st = proj.factory.call_state(fn.addr)
    st.inspect.b('address_concretization', when=angr.BP_AFTER, action=address_breakfun)
    sm = proj.factory.simulation_manager(st)
    sm.run()
    for k, v in replacement_dict.items():
        print(f"{hex(k)}: {v}")
    print(bases_dict)
    conts = ""
    for st in sm.deadended:
        conts += str(st.solver.constraints)
    for k, v in replacement_dict.items():
        conts = re.sub(f"(0x|mem_){format(k, 'x')}[_0-9]*", v, conts)
    print(conts)


if __name__ == '__main__':
    main()
