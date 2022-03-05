import angr
import os
import claripy
import inspect
import pickle
import re
import time
import logging
import json
import argparse
import itertools
from glob import glob

def get_cfg_funcs(proj, binary, excluded):
    """
    get functions that are suitable for analysis, (funcs that are defined in the binary and not libc funcs...)
    """
    return list(filter(None, [f if f.binary_name == binary and (not f.is_plt) and not f.name.startswith(
        "sub_") and not f.name.startswith("_") and f.name not in excluded else None for f in
                              proj.kb.functions.values()]))


def analyze_func(proj, fun, cfg):
    print(f"started running {fun.name}")
    call_state = proj.factory.call_state(fun.addr, add_options={
        'CALLLESS': True, 'NO_SYMBOLIC_SYSCALL_RESOLUTION': True, 'TRACK_ACTION_HISTORY': True , 'TRACK_CONSTRAINTS': True
    })
    # dropped the relativization in the last moment due to time consedirations, and we think that the address_breakfun


  # need to be checked again...
    # call_state.inspect.b('address_concretization', when=angr.BP_AFTER, action=address_breakfun)
    sm = proj.factory.simulation_manager(call_state)
    sm.use_technique(angr.exploration_techniques.LoopSeer(cfg=cfg, bound=2)) 
    #print("--------------------------------------")
    sm1=sm
    while (True):
        if hasattr(sm,'deadended') and len(sm.deadended) > 0:
            break
        else:
            sim = sm.step()
            if len(sim.active) > 0:
                curr = sim.active[0]
                print(curr.solver.constraints)
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
    sm1.run()
    return sm1

def block_to_ins(block: angr.block.Block):
    result = []
    for ins in block.capstone.insns:
        print(ins)
        op_str = ins.op_str
        operands = op_str.strip(" ").split(",")
        operands = [i.strip().replace("[","").replace("]", "") for i in operands if i != ""]
        parsed_ins = [ins.mnemonic] + list(filter(None, operands))
        result.append("|".join(parsed_ins).replace(" ", "|") + "|\t")
        # result.append(f"{ins.mnemonic}|{operands[0]}|{operands[1]}".replace(" ", "|"))
    return "|".join(result)

def run(binary_name):
    proj = angr.Project(binary_name)
   # print(proj.factory.path_group())
    cfg = proj.analyses.CFGFast()
    s = claripy.Solver()
    idfer = proj.analyses.Identifier()
    for funcInfo in idfer.func_info:
        print(funcInfo.name)    
    funcs = get_cfg_funcs(proj, binary_name, {})
    for test_func in funcs:
        sm: angr.sim_manager.SimulationManager = analyze_func(proj, test_func, cfg)
        sashes = sm.stashes.values()
        print("/******ALL CONSTRAINTS !!***********************/")
        for exec_paths in sashes:
            for exec_path in exec_paths:
                #print(exec_path)
                #print("printing func")
                #print(test_func)
                #print("printing constraint")
                #print(exec_path.libc)
                #print(exec_path.regs.r16)
                blocks = [proj.factory.block(baddr) for baddr in exec_path.history.bbl_addrs]
                processsed_code = "|".join(list(filter(None, map(block_to_ins, blocks))))
                #print(processsed_code)
                print(exec_path.log.actions.gi_frame)
                print(exec_path.log.actions.__class__.__dict__.items())
                for p in exec_path.se.constraints:
                    print((p))
                   # print(type(exec_paths))
                    #print(type(exec_path))
                #print("Heyyy /**/")
                #print(exec_path.solver)
                    #print(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_name", type=str, required=True)
    args = parser.parse_args()
    run(args.binary_name)

