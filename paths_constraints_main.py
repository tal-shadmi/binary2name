from sym_graph import *
from typing import Dict, Any, List, Callable, Set


from angr import Project
from angr.exploration_techniques.loop_seer import LoopSeer
from angr.block import Block
from angr.sim_manager import SimulationManager
from claripy import ast
import os
import re
import time
import logging
import json
import argparse
import itertools
from glob import glob

import logging
import resource
import sys

bases_dict = dict()
replacement_dict = dict()
start_time = 0


# REPR = representation
        
def time_limit_check(simulation_manager):
    global start_time
    minutes_limit = 10
    should_stop = time.time() - start_time > (60 * minutes_limit)
    if should_stop:
        print("stopped exploration")
    return should_stop

# Analyze a specific function with angr
# proj is the project object, cfg IS THE ACTUAL CONTROL-FLOW GRAPH
def analyze_func(proj, bin_func, cfg):
    print(f"started running {bin_func.name}")
    call_state = proj.factory.call_state(bin_func.addr, add_options={
        'CALLLESS': True, 'NO_SYMBOLIC_SYSCALL_RESOLUTION': True
    })
    sm = proj.factory.simulation_manager(call_state)  # Creates a simulation manager, ready to start from the specific function
    sm.use_technique(LoopSeer(cfg=cfg, bound=2))
    
    global start_time
    start_time = time.time()
    sm.run(until=time_limit_check)
    print(f"finished {bin_func.name}")
    return sm


def get_cfg_funcs(proj, binary, excluded):
    """
    get functions that are suitable for analysis, (funcs that are defined in the binary and not libc funcs...)
    """
    return list(filter(None, [f if f.binary_name == binary and (not f.is_plt) and not f.name.startswith(
        "sub_") and not f.name.startswith("_") and f.name not in excluded else None for f in
                              proj.kb.functions.values()]))


def block_to_ins(block: Block):
    result = []
    for ins in block.capstone.insns:
        op_str = ins.op_str
        operands = op_str.strip(" ").split(",")
        operands = [i.strip().replace("[","").replace("]", "") for i in operands if i != ""]
        parsed_ins = [ins.mnemonic] + list(filter(None, operands))
        result.append("|".join(parsed_ins).replace(" ", "|") + "|    ")
    return "|".join(result)


def remove_consecutive_pipes(s1):
    s1 = re.sub("(\|(\s)+\|)", "|", s1)
    return re.sub("(\|)+", "|", s1)


# def constraint_to_str(con, replace_strs=[', ', ' ', '(', ')'], max_depth=100):
#     repr = con.shallow_repr(max_depth=max_depth, details=con.MID_REPR).replace('{UNINITIALIZED}', '')
#     repr=re.sub("Extract\([0-9]+\, [0-9]+\,","",repr)
#     for r_str in replace_strs:
#         repr = repr.replace(r_str, '|')

#     return remove_consecutive_pipes(repr) + "    "

def constraint_to_str(constraint: ast.Base, max_depth: int=100) -> str:
    return constraint.shallow_repr(max_depth=max_depth, details=constraint.MID_REPR).replace('{UNINITIALIZED}', '')


def gen_new_name(old_name):
    if re.match(r"mem", old_name):
        return 'mem_%s' % old_name.split('_')[2]
    if re.match(r"fake_ret_value", old_name):
        return 'ret'
    if re.match(r"reg", old_name):
        return re.sub("(_[0-9]+)+", '', old_name)
    if re.match(r"unconstrained_ret", old_name):
        return re.sub("(_[0-9]+)+", '', old_name[len("unconstrained_ret_") : ])
    return old_name


def varify_constraints(constraints, variable_map=None, counters=None, max_depth=8):
    """
    abstract away constants from the constraints
    @param constraints is raw constraints from angr (eg. node.recent_constraints)
    """
    #counters = {'mem': itertools.count(), 'ret': itertools.count()} if counters is None else counters
    variable_map = {} if variable_map is None else variable_map  # Variable map contains a mapping between old and new variable names
    new_constraints = []  # constraints after name changed and simplified
    variable_map['Extract'] = ""

    m = None  # m used as counter / holder for number added to "mem" variables.
    for constraint in constraints:
        if constraint.concrete:  # need to figure out what concrete means.
            continue
        for variable in constraint.leaf_asts():  # returns iterator over the leaves of the AST
            if variable.op in { 'BVS', 'BoolS', 'FPS' }:  # Generate new name if variable op needs it
                new_name = gen_new_name(variable.args[0])
                if re.match(r"mem", new_name):
                    if m is None :
                        m = int(new_name.split('_')[1])
                    else:
                        m = min(m,int(new_name.split('_')[1]))
                variable_map[variable.cache_key] = variable._rename(new_name)  # preparing variable_map for name swapping in the line 123
        '''
            converting constraint to string after renaming all the necessary vars with the variable_map we build along the way.
            look into constraint_to_str to understand further simplifying done in there too.
            max depth is the maximum fold of the ast (consider enlarging)
        '''
        new_constraints.append(constraint_to_str(constraint.replace_dict(variable_map), max_depth=max_depth))

    final_constraints = []  # initializing new list of even further simplified constraints.
    if m is not None:  # meaning we found a variable inside a constraint that accessed memory
        for constraint in new_constraints :  # iterate over already simplified constraints
            split = constraint.split("|")  # split to tokens
            for i,s in enumerate(split):  # enumerating tokens (WHY?)
                if re.match(r"mem", s):
                    new_s = 'mem_%d' % (int(s.split('_')[1]) -m)
                    constraint = constraint.replace(s,new_s)
            final_constraints.append(constraint)
    # basically we iterate over all the constraints including the phrase mem_%NUM% and normalize the number to start from 0.
    else:
        final_constraints = new_constraints
    return variable_map, final_constraints  # return simplified constraints and variable map for further use.
    '''
    notes:
    1. using variable_map correctly here is crucial. or else all the memory counters wouldn't be synchronized and we will get garbage insterted into the system
    and worse then that, we can get two simplified names to the same variable!! (VERY BAD) 
    2. need to understand if keeping all constraints raw is really the right thing or not.
    3. if constructing the method from 0, need to think about how it will affect the calls to it (considering right use of variable_map in older calls)
    '''


# Remove the Numbers from the function names + tokenize the function name.
def tokenize_function_name(function_name):
    name = "".join([i for i in function_name if not i.isdigit()])
    return "|".join(name.split("_"))



def generate_dataset(train_binary: str, output_dir: str, dataset_name: str, no_usables_file: bool):
    """
    The main preprocessing fuction. Performs the analysis of each function serially.
    In our implementation, paralellism is in binary-file granularity.

        :param train_binary: the binary file to preprocess.
        :type train_binaries: str
        :param output_dir: the name of the output directory (will be put under preprocessed_data)
        :type output_dir: str
        :param dataset_name: the name of the binary dataset (will be searched under our_dataset)
        :type dataset_name: str
        :param no_usables_file: indicates there is no usable_functions_names.txt file, so all functions will be analyzed.
        :type no_usables_file: bool
    """
    # Generate a "function usable" predicate
    # If no usable_function_names file exists, accept all functions.
    is_function_usable = (lambda _: True)
    if not no_usables_file:
        usable_functions_file = open("our_dataset/" + dataset_name + "/usable_functions_names.txt", "r")
        usable_functions = [name.strip() for name in usable_functions_file]
        is_function_usable = (lambda x: x in usable_functions)
    
    # Check which functions have already been analyzed
    output_dir = f"preprocessed_data/{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Disabling this part of the algorithm, since it eats up memory and is not implemented well
    # analyzed_funcs = get_analyzed_funcs(output_dir)
    analyzed_funcs = set()

    analyze_binary(analyzed_funcs, train_binary, output_dir, is_function_usable)



def analyze_binary(analyzed_funcs: Set[str], binary_name: str, output_dir: str, is_function_usable: Callable[[str], bool]):
    excluded = {'main', 'usage', 'exit'}.union(analyzed_funcs)

    proj = Project(binary_name, auto_load_libs=False) # Load angr project and calculate CFG
    cfg = proj.analyses.CFGFast()  # cfg is the ACTUAL control-flow graph

    binary_name = os.path.basename(binary_name) # Make the output directory for this binary
    binary_output_dir = os.path.join(output_dir, f"{binary_name}")
    os.makedirs(binary_output_dir, exist_ok=True)

    funcs = get_cfg_funcs(proj, binary_name, excluded)
    print(f"{binary_name} have {len(funcs)} funcs")
    for test_func in funcs:        
        if (test_func.name in analyzed_funcs) or not is_function_usable(tokenize_function_name(test_func.name)): # Skip unusable/computed functions
            print(f"skipping {tokenize_function_name(test_func.name)}")
            continue
        print(f"analyzing {binary_name}/{test_func.name}")
        
        analyzed_funcs.add(test_func.name)
        try:
            sm = analyze_func(proj, test_func, cfg) # Perform Carol's angr analysis
            with open(os.path.join(binary_output_dir, f"{test_func.name}.json"), "w") as output:
                sm_to_graph(sm, output, test_func.name) # calculate the constraint-full CFG
        except Exception as e:
            logging.error(str(e))
            logging.error(f"got an error while analyzing {test_func.name}")
        
    return analyzed_funcs


def get_analyzed_funcs(dataset_path: str) -> Set[str]:
    binaries = os.scandir(dataset_path)
    analyzed_funcs = set()
    for entry in binaries:
        funcs = glob(f"{entry.path}/*")
        analyzed_funcs.update(map(lambda x: x[:-len(".pkl")] if x.endswith(".pkl") else x, map(os.path.basename, funcs)))

    return analyzed_funcs


def find_target_constants(line):
    targets_mapper = {}
    targets_counter = itertools.count()
    
    found_targets = set(re.findall(r"jmp\|0[xX][0-9a-fA-F]+|jnb\|0[xX][0-9a-fA-F]+|jnbe\|0[xX][0-9a-fA-F]+|jnc\|0[xX][0-9a-fA-F]+|jne\|0[xX][0-9a-fA-F]+|jng\|0[xX][0-9a-fA-F]+|jnge\|0[xX][0-9a-fA-F]+|jnl\|0[xX][0-9a-fA-F]+|jnle\|0[xX][0-9a-fA-F]+|jno\|0[xX][0-9a-fA-F]+|jnp\|0[xX][0-9a-fA-F]+|jns\|0[xX][0-9a-fA-F]+|jnz\|0[xX][0-9a-fA-F]+|jo\|0[xX][0-9a-fA-F]+|jp\|0[xX][0-9a-fA-F]+|jpe\|0[xX][0-9a-fA-F]+|jpo\|0[xX][0-9a-fA-F]+|js\|0[xX][0-9a-fA-F]+|jz\|0[xX][0-9a-fA-F]+|ja\|0[xX][0-9a-fA-F]+|jae\|0[xX][0-9a-fA-F]+|jb\|0[xX][0-9a-fA-F]+|jbe\|0[xX][0-9a-fA-F]+|jc\|0[xX][0-9a-fA-F]+|je\|0[xX][0-9a-fA-F]+|jz\|0[xX][0-9a-fA-F]+|jg\|0[xX][0-9a-fA-F]+|jge\|0[xX][0-9a-fA-F]+|jl\|0[xX][0-9a-fA-F]+|jle\|0[xX][0-9a-fA-F]+|jna\|0[xX][0-9a-fA-F]+|jnae\|0[xX][0-9a-fA-F]+|jnb\|0[xX][0-9a-fA-F]+|jnbe\|0[xX][0-9a-fA-F]+|jnc\|0[xX][0-9a-fA-F]+|jne\|0[xX][0-9a-fA-F]+|jng\|0[xX][0-9a-fA-F]+|jnge\|0[xX][0-9a-fA-F]+|jnl\|0[xX][0-9a-fA-F]+|jnle\|0[xX][0-9a-fA-F]+|jno\|0[xX][0-9a-fA-F]+|jnp\|0[xX][0-9a-fA-F]+|jns\|0[xX][0-9a-fA-F]+|jnz\|0[xX][0-9a-fA-F]+|jo\|0[xX][0-9a-fA-F]+|jp\|0[xX][0-9a-fA-F]+|jpe\|0[xX][0-9a-fA-F]+|jpo\|0[xX][0-9a-fA-F]+|js\|0[xX][0-9a-fA-F]+|jz\|0[xX][0-9a-fA-F]+ ", line))
    for target in found_targets:
        print("removing targets")
        target = re.sub("[a-z]+\|", "" , target)
        if target not in targets_mapper:
            targets_mapper[target] = f"target_{next(targets_counter)}"
    for target, replacement in sorted(targets_mapper.items(), key=lambda x: len(x[0]), reverse=True):
                line = line.replace(target, replacement)
    return line


#--------------------- ITTAY AND ITAMAR'S CODE---------------------#


def varify_constraints_raw(constraints) -> List[str]:
    """
    Performs minimal parsing of the constraints into a string.
    """
    new_constraints = []
    for constraint in constraints:
        if constraint.concrete:
            continue
        new_constraints.append(constraint_to_str(constraint))
    
    return new_constraints


def address_to_content_raw(proj: Project, baddr: int):
    full_block = proj.factory.block(baddr)
    raw_instructions = block_to_ins(full_block)
    return raw_instructions


def address_to_content(proj: Project, baddr: int):
    raw_instructions = address_to_content_raw(proj, baddr)
    instructions = re.sub("r[0-9]+", "reg", raw_instructions)
    instructions = re.sub("r[0-9]+", "reg", instructions)
    instructions = re.sub("xmm[0-9]+", "xmm", instructions)
    instructions = find_target_constants(instructions)
    return instructions


def sm_to_graph(sm: SimulationManager, output_file, func_name):
    proj = sm._project
    final_states_lists = filter(None, sm.stashes.values())

    # TODO: make sure you want to treat the "deadended" and "spinning" states the same way
    final_states = [item for sublist in final_states_lists for item in sublist]
    assert(final_states is not []) # assert that final states list is not empty else we dont have what to work with
    # compose all routs backtracking from final to initial
    all_paths = []
    for state in final_states:
        current_node = state.history
        state_path = [("loopSeerDum", current_node.recent_constraints)]
        while current_node.addr is not None:
            state_path.insert(0, (current_node.addr, (current_node.parent.recent_constraints if current_node.parent else [])))
            current_node = current_node.parent
        all_paths.append(state_path)

    # find the root and assert it is equal for all
    initial_node = all_paths[0][0]
    for path in all_paths:
        assert(path[0][0] == initial_node[0])  # WARNING: very redundent, only checking adress
        assert(path[0][1] == [])  # assert all root's contain no constraints as expected

    root = Vertex(initial_node[0], address_to_content(proj, initial_node[0]))
    sym_graph = SymGraph(root, func_name)

    # In each iteration, add a new constrainted vertex to the graph and connect it to the previous vertex.
    # In the SymGraph, vertex addition handles multiple constraint options and adds an OR relation.

    for path in all_paths:
        prev = root
        for i in range(1, len(path)):
            constraint_list = varify_constraints_raw(path[i][1])
            if type(path[i][0]) == str:
                dst = Vertex(path[i][0], "no_instructions", ["|".join(constraint_list)])
            else:
                dst = Vertex(path[i][0], address_to_content_raw(proj, path[i][0]), ["|".join(constraint_list)])
            sym_graph.addVertex(dst)
            edge = Edge(prev.baddr, dst.baddr)
            sym_graph.addEdge(edge)
            prev = dst
    
    our_json = sym_graph.__str__()
    our_json = our_json.replace("'", "\"").replace("loopSeerDum", "\"loopSeerDum\"")
    parsed = json.loads(our_json)
    to_write = json.dumps(parsed, indent=4, sort_keys=True)
    output_file.write(to_write)
    
#--------------------- ITTAY AND ITAMAR'S CODE END---------------------#


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_idx", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mem_limit", type=int, required=True)
    parser.add_argument("--no_usables_file", dest="no_usables_file", action="store_true")
    args = parser.parse_args()

    logging.getLogger('angr').setLevel('CRITICAL')  # Silence angr  
    heap_resource = resource.RLIMIT_DATA  # Limit data capture
    # soft_l, hard_l = resource.getrlimit(heap_resource)
    resource.setrlimit(heap_resource, (args.mem_limit*2**30, (args.mem_limit+5)*2**30))
    sys.setrecursionlimit(10**6)  # Limit stack

    


    binaries = os.listdir("our_dataset/" + args.dataset)
    binaries.sort()
    binaries = [f"our_dataset/{args.dataset}/{binary}" for binary in binaries]

    generate_dataset(binaries[args.binary_idx], args.output, args.dataset, args.no_usables_file)


if __name__ == '__main__':
    main()


# python3 paths_constraints_main.py --dataset nero_ds/TRAIN --output nero_train_out --binary_idx 0 --no_usables_file > main_log.txt 2>&1 &
