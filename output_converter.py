import os
import shutil
import json
from typing import List, Dict, Tuple
from jsonpickle import encode
import argparse
from tqdm import tqdm
import random
import re
import multiprocessing
#import multiprocess
#import pebble
import datetime
import math
import pprint

TIMEOUT_PER_FILE = 3
CONSTRAINT_DELIM = '|'
OUR_API_TYPE = 'F'  # Meaningless - simply here to notify this is not a NORMAL_PROC or INDIRECT_PROC in the Nero Preprocessing.

MEM_DIFF_THRESH = 20
RET_DIFF_THRESH = 20
CONVERTED_DS_PREFIX = 'Converted_'
SYM_EXE_MAX_OUTPUT_TO_PROCESS = 1000000


def is_num(val: str) -> bool:
    return val.startswith('0x') or re.match('[0-9]+', val) != None


def is_mem(val: str) -> bool:
    return 'mem' in val


def is_reg(val: str) -> bool:
    return 'reg' in val


def is_retval(val: str) -> bool:
    return 'fake_ret_value' in val


def collect_to_file(file_list: List[str], filename: str) -> None:
    collective_files = ''
    # --------------------- TAL'S CODE START---------------------#
    #num_files_found_in_model_train_set = 0
    #if 'test.json' in filename:
        #with open(os.path.join('../nero/nero', 'model_train.json'), 'r') as model_train_set:
            #for function_file in file_list:
                #function_name = function_file.split('/')[2].replace('.json', '')
                #if function_name not in model_train_set.read():
                    #with open(function_file, 'r') as file:
                        #collective_files += file.read() + '\n'
                #else:
                    #num_files_found_in_model_train_set += 1
                #model_train_set.seek(0)
        #print("found {} functions in model train set".format(num_files_found_in_model_train_set))
    #else:
        #for function_file in file_list:
            #with open(function_file, 'r') as file:
                #collective_files += file.read() + '\n'
    
    if 'train.json' in filename:
        binaries = [b for b in os.listdir('../nero_dataset_binaries/TRAIN') if os.path.isfile(os.path.join('../nero_dataset_binaries/TRAIN', b))]
    if 'test.json' in filename:
        binaries = [b for b in os.listdir('../nero_dataset_binaries/TEST') if os.path.isfile(os.path.join('../nero_dataset_binaries/TEST', b))]
    if 'validation.json' in filename:
        binaries = [b for b in os.listdir('../nero_dataset_binaries/VALIDATE') if os.path.isfile(os.path.join('../nero_dataset_binaries/VALIDATE', b))]
    for function_file in file_list:
        binary_name = function_file.split('/')[1]
        if binary_name in binaries:
            with open(function_file, 'r') as file:
                collective_files += file.read() + '\n'
    # --------------------- TAL'S CODE END---------------------#
    #for function_file in file_list:
        #with open(function_file, 'r') as file:
            #collective_files += file.read() + '\n'

    with open(os.path.join('../ready_data', filename), 'w') as file:
        file.write(collective_files)


def separate_arguments(args):
    arguments = []
    delimiter_count = 0
    begin_index = 0
    end_index = 0

    while end_index < len(args):
        letter = args[end_index]
        if letter == '(':
            delimiter_count += 1
        if letter == ')':
            delimiter_count -= 1
        if letter == ',' and delimiter_count == 0:
            arguments.append(args[begin_index:end_index])
            begin_index = end_index + 2  # (, )
            end_index = begin_index
        end_index += 1

    arguments.append(args[begin_index:])
    if delimiter_count != 0:
        print("delimiter_count", delimiter_count)
        print("args", args)
        print('Warning! delimiters are not equal on both sides, check for inconsistencies')
        print('arguments', arguments)
        raise Exception
    return arguments


def dissolve_function_call(str_call):
    delimiter_open = str_call.find('(')
    delimiter_close = str_call.rfind(')')
    arguments = separate_arguments(str_call[delimiter_open + 1:delimiter_close])
    call_name = str_call[:delimiter_open]
    return call_name, arguments


def convert_argument(argument: str) -> tuple:
    line_split = argument.split('_')
    if is_mem(argument):
        argument_type = 'MEMORY'
        argument = 'mem'
    elif is_reg(argument):
        argument_type = 'REGISTER'
        #argument = 'reg' + line_split[1]
        argument = 'reg'
    elif is_num(argument):
        argument_type = 'CONSTANT'
    elif is_retval(argument):
        argument_type = 'RET_VAL'
        argument = 'fake_ret'
    elif argument.startswith(OUR_API_TYPE):
        argument_type = 'FUNCTION_CALL'
    #elif argument.startswith("EAX"):
    #    argument_type = 'REGISTER'
    #    argument = 'EAX'
    else:
        argument_type = 'UNKNOWN'
    return argument_type, argument


class ConstraintAst:
    def __init__(self, value='dummy_name', children: List['ConstraintAst'] = []):
        self.value = value
        self.children = children

    def remove_filler_nodes(self, bad_name: str, argnum: int) -> None:
        if self.value == bad_name:
            if len(self.children) >= argnum:
                self.value = self.children[argnum - 1].value
                self.children = self.children[argnum - 1].children
            else:
                assert len(self.children) == 0

        if self.children is None:  # if the grandchild is none, meaning the son which replaced the father is a leaf
            return

        for child in self.children:
            child.remove_filler_nodes(bad_name, argnum)

    def __export_ast_to_list(self) -> List:
        list_repr = []
        if not self.children:
            return []

        for child in self.children:
            list_repr += child.__export_ast_to_list()

        my_call = (self.value, [child.value for child in self.children])
        list_repr.append(my_call)
        return list_repr

    def convert_list_to_nero_format(self) -> List:
        """
        convert all function calls existing in the list into the nero format.
        we roll the list, popping from the start, converting then appending to the end
        because we do that len(list) times, there !should! be no problems...
        """
        constraints_as_calls = self.__export_ast_to_list()
        for i in range(len(constraints_as_calls)):
            func_name, arguments = constraints_as_calls.pop(0)
            function_call = [func_name]
            for arg in arguments:
                function_call.append(convert_argument(arg))
            converted_function_call = tuple(function_call)
            constraints_as_calls.append(converted_function_call)
        return constraints_as_calls


def are_constraints_similar(first: ConstraintAst, second: ConstraintAst) -> bool:
    if len(first.children) != len(second.children):
        return False
    
    if first.children == [] and second.children == []:
        if first.value == second.value:
            return True

        if is_num(first.value) and is_num(second.value):
            return True

        elif is_mem(first.value) and is_mem(second.value):
            split_a = [int(x, 16) for x in first.value.split('_')[1:]]
            split_b = [int(x, 16) for x in second.value.split('_')[1:]]
            if split_a[0] != split_b[0]:    # address
                return False
            if abs(split_a[1] - split_b[1]) > MEM_DIFF_THRESH:  # arbitary id (incremented sequentially)
                return False
            if abs(split_a[2] != split_b[2]) > MEM_DIFF_THRESH:  # size
                return False
            return True
        
        elif is_retval(first.value) and is_retval(second.value):
            ret_a = int(first.value.split('_')[-2], 16)
            ret_b = int(second.value.split('_')[-2], 16)
            if abs(ret_a - ret_b) > RET_DIFF_THRESH:
                return False
            return True
        
        else:
            return False
    
    if first.value != second.value:
        return False
    for child_a, child_b in zip(first.children, second.children):
        if not are_constraints_similar(child_a, child_b):
            return False
    
    return True


def are_constraints_contradicting(first: ConstraintAst, second: ConstraintAst) -> bool:
    contradicting = [('eq', 'ne')]
    if (first.value, second.value) in contradicting or (second.value, first.value) in contradicting:
        return all([are_constraints_similar(child_a, child_b) for child_a, child_b in zip(first.children, second.children)])
    return False


def merge_constraints_similar(first: ConstraintAst, second: ConstraintAst) -> ConstraintAst:
    assert len(first.children) == len(second.children)
    value = first.value
    if first.value != second.value:
        if is_num(first.value) and is_num(second.value):
            value = '0x?'
        elif is_mem(first.value) and is_mem(second.value):
            value = "mem_?"
        elif is_retval(first.value) and is_retval(second.value):
            value = "fake_ret_value_?_?"
        else:
            value = '?'
        
    children = [merge_constraints_similar(child_a, child_b) for child_a, child_b in zip(first.children, second.children)]
    return ConstraintAst(value, children)


def get_constraint_ast(constraint: str, curr_depth: int, max_depth: int) -> ConstraintAst:
    constraint_ast = ConstraintAst(children=[])
    function_name, arguments = dissolve_function_call(constraint)
    function_name = OUR_API_TYPE + function_name
    constraint_ast.value = function_name
    if curr_depth >= max_depth:
        return constraint_ast
    for arg in arguments:
        if '(' in arg or ')' in arg:
            constraint_ast.children.append(get_constraint_ast(arg, curr_depth + 1, max_depth))
        else:
            constraint_ast.children.append(ConstraintAst(value=arg))
    return constraint_ast


class OutputConvertor:
    def __init__(self, dataset_name: str, sample_path: int, sample_constraint: int):
        self.filenames = []
        self.converted_filenames = []
        self.src = dataset_name
        self.dest = CONVERTED_DS_PREFIX + dataset_name
        self.sample_path = sample_path
        self.sample_constraint = sample_constraint

    def backup_all_files(self):
        """
        update the self.filename list to contain all the files in the given dataset
        we presume the given dataset is a folder in the same directory as the script
        we copy the dataset first to a different name directory so working on it will not harm
        the previous model.
        """
        src = self.src
        dest = self.dest
        if os.path.isdir(dest):
            print('converted dir already exists, removing')
            shutil.rmtree(dest)
        
        print('Started copying dataset for backup')
        shutil.copytree(src, dest)
        print('Finished backup, starting to scan files')

    def clear_converted_dataset(self):
        src = self.src
        dest = self.dest
        if os.path.isdir(dest):
            print('converted dir already exists, removing')
            shutil.rmtree(dest)
        
        
    def load_all_files(self):
        dataset_name = self.src
        bin_folders = list(
            map(lambda x: os.path.join(dataset_name, x) if x[-4:] != '.txt' else None, os.listdir(dataset_name)))
        bin_folders = list(filter(None, bin_folders))

        for path in bin_folders:
            self.filenames += list(map(lambda x: os.path.join(path, x), os.listdir(path)))

        for file in self.filenames:
            if not file.endswith('.json'):
                self.filenames.remove(file)
        print('Finished scanning and adding all files\n', 'added {} files'.format(len(self.filenames)))


    def convert_dataset(self):
        print(datetime.datetime.now().strftime("%H:%M:%S"), 'Starting to convert json files')
        i = 0
        pool = multiprocessing.Pool(32)
        for success, converted_filename in pool.imap_unordered(self.convert_json, self.filenames, chunksize = 1000):
            if success:
                self.converted_filenames.append(converted_filename)
            i = i + 1
            if i % 100 == 0:
                p = int(i*100/(1+len(self.filenames)))
                print(f'{p}% processed.', end = "    \r")
            
        print('\n', datetime.datetime.now().strftime("%H:%M:%S"), 'Done converting, data should be ready')
        print('{} out of {} files were converted which mean they were not empty or too large.'.format(len(self.converted_filenames), len(self.filenames)))


    def get_stats(self):
        print(datetime.datetime.now().strftime("%H:%M:%S"), 'Starting to calculate stats')
        i = 0
        pool = multiprocessing.Pool(32)
        precent_block_constraints = {}
        num_files_with_json_error = 0
        for success, result in pool.imap_unordered(self.get_stats_json, self.filenames, chunksize = 1000):
            if success:
                if result["error_parsing_json"] == True:
                    num_files_with_json_error = num_files_with_json_error + 1
                    continue
                per = math.ceil(result["precent_block_constraints"]*10)
                precent_block_constraints[per] = precent_block_constraints.get(per, 0) + 1
            i = i + 1
            if i % 100 == 0:
                p = int(i*100/(1+len(self.filenames)))
                print(f'{p}% processed.', end = "    \r")
        for per, value in sorted(precent_block_constraints.items()):
            print(f"{(per-1)*10}%-{per*10}% -- {math.ceil((value/i)*100)}%")
        pprint.pprint(precent_block_constraints)
        print("num_files_with_json_error", num_files_with_json_error)
        print('\n', datetime.datetime.now().strftime("%H:%M:%S"), 'Done calculating stats')

    def get_stats_json(self, filename):
        filesize = os.path.getsize(filename)
        result = {}
        if filesize == 0 or filesize > SYM_EXE_MAX_OUTPUT_TO_PROCESS:
            return False, None
        result["error_parsing_json"] = False
        try:
            with open(filename, 'r') as function_file:
                data = json.load(function_file)
        except Exception as e:
            # print (str(e))
            # print(filename)
            #os.remove(filename)
            result["error_parsing_json"] = True
            return True, result
        
        nodes = data['GNN_DATA']['nodes']
        num_nodes_with_constraints = 0
        num_nodes = 0
        for node in nodes:
            node_has_constarints = False
            for c in node["constraints"]:
                if len(c[1]) > 0:
                    node_has_constarints = True
                    break
            num_nodes = num_nodes + 1
            num_nodes_with_constraints = num_nodes_with_constraints + int(node_has_constarints)
        result["precent_block_constraints"] = num_nodes_with_constraints/num_nodes
        return True, result


    def __convert_edges(self, edges: List) -> List:
        converted_edges = []
        for edge in edges:
            new_edge = (edge['src'], edge['dst'])
            converted_edges.append(new_edge)
        return converted_edges

    def __process_constraints_to_asts(self, block_constraints: List[str]) -> List[ConstraintAst]:
        with open('conversion_config.json', 'r') as config_file:
            data = json.load(config_file)
            MAX_TOKENS_PER_CONSTRAINT = data['MAX_TOKENS_PER_CONSTRAINT']

        filtered_asts = []
        for path_constraints_string in block_constraints:
            constraints = path_constraints_string.split(CONSTRAINT_DELIM)
            for constraint in constraints:
                # get the constraint AST
                constraint_ast = get_constraint_ast(constraint, 1, 3) #return constraints tree upto a depths of max_depth=3
                # filter all unwanted functions
                constraint_ast.remove_filler_nodes(OUR_API_TYPE + 'Extract', 3)
                constraint_ast.remove_filler_nodes(OUR_API_TYPE + 'ZeroExt', 2)
                # constraint_ast.remove_filler_nodes(OUR_API_TYPE + 'invert', 1)
                # constraint_ast.remove_filler_nodes(OUR_API_TYPE + 'Concat', 1)  # Random choice - perhaps a different choice would be better.

                filtered_asts.append(constraint_ast)

        return filtered_asts  # TODO: Use the MAX_TOKENS parameter to cut the list according the the rules...

    def __prettify_constraints(self, block_constraints: List[str]) -> List[str]:
        """
        goals: take out garbage from the constraint like BOOL __X__
        strip ' ' and '<' and '>'
        structure of block_constraints:
        It is a list of strings.
        Each string represents the constraints of a single path through the CFG.
        The constraints of each single path are delimited with a '|' character.
        """
        converted_block_constraints = []
        for path_constraints in block_constraints:
            # print(path_constraints)
            # print(type(path_constraints))
            converted_path_constraints = []
            for constraint in path_constraints.split('|'):
                # Remove the <Bool ... > prefix and suffix of each constraint.
                converted_constraint = constraint.replace('Bool', '').replace('<', '').replace('>', '').replace('BV32 ', '').strip()
                # Clean the representation of boolean ops: remove the '__' prefix and suffix.
                converted_constraint = re.sub(r'__(?P<op>[a-zA-Z]+)__', r'\g<op>', converted_constraint)
                converted_path_constraints.append(converted_constraint)
            # Style back to the original format
            converted_block_constraints.append('|'.join(converted_path_constraints))
        return converted_block_constraints

    def __reduce_constraints(self, block_constraints: List[str]) -> List[str]:
        """
        goals: reduce the number of constraints so the model would like more easily
        """
        converted_block_constraints = []
        filtered_block_constraints = list(filter(lambda c: len(c[1]) > 0, block_constraints)) #filter paths that have zero constraints
        if len(filtered_block_constraints) == 0:
            return ['']
        # print("block_constraints", len(block_constraints), block_constraints)
        paths_len_and_constraints = random.sample(filtered_block_constraints, min(len(filtered_block_constraints), self.sample_path))
        # print("paths_len_and_constraints", len(paths_len_and_constraints), paths_len_and_constraints)
        for path_len, path_constraints in paths_len_and_constraints: # path_len is the length of the execution path (until the current block) that contibuted these
            selected_path_constraints = random.sample(path_constraints, min(len(path_constraints), self.sample_constraint))
            converted_block_constraints.extend(selected_path_constraints)
            # print("converted_block_constraints", len(converted_block_constraints), converted_block_constraints)
        # print("|".join(converted_block_constraints))
        return ["|".join(converted_block_constraints)]


    # --------------------- TAL'S CODE START---------------------#
    # function to manually deduct constraints to a certain number
    def __deduct_constraints(self, block_constraints: List[str], num_paths_after_deduction: int) -> List[str]:
        """
        get block constraints and return those constraints after deduction. the number of paths will be
        num_paths_after_deduction long.
        """
        print("this is the original version of block constraints with " + str(len(block_constraints)) + " paths:")
        for constraint in block_constraints:
            print(constraint)
        paths_counter = 0
        deducted_block_constraints = []
        for path_constraints in block_constraints:
            deducted_path_constraints = []
            paths_counter = paths_counter + 1
            if paths_counter > num_paths_after_deduction:
                break
            for constraint in path_constraints.split('|'):
                deducted_path_constraints.append(constraint)
            deducted_block_constraints.append('|'.join(deducted_path_constraints))
        print("this is a deducted version of block constraints because " + str(paths_counter) + " paths left")
        for constraint in deducted_block_constraints:
            print(constraint)
        return deducted_block_constraints

    # --------------------- TAL'S CODE END---------------------#

    # This algorithm is rudimentary at best.
    # Feel free to make it more efficient :)
    def __deduplicate_constraints(self, constraint_asts: List[ConstraintAst]) -> List[ConstraintAst]:
        i = 0
        while i < len(constraint_asts):
            duplicated = False
            merged_ast = None
            contradicting = False
            j = i + 1
            while j < len(constraint_asts) and not contradicting:
                if are_constraints_similar(constraint_asts[i], constraint_asts[j]):
                    if not duplicated:
                        merged_ast = merge_constraints_similar(constraint_asts[i], constraint_asts[j])
                        duplicated = True
                    constraint_asts.pop(j)
                elif are_constraints_contradicting(constraint_asts[i], constraint_asts[j]):
                    constraint_asts.pop(j)
                    constraint_asts.pop(i)
                    contradicting = True
                else:
                    j += 1
            if not contradicting:
                if duplicated:  # Replace the original with the generalization
                    constraint_asts[i] = merged_ast
                i += 1
        return constraint_asts

    def __convert_nodes(self, nodes: List) -> Dict:
        with open('conversion_config.json', 'r') as config_file:
                data = json.load(config_file)
                MAX_TOKENS_PER_CONSTRAINT = data['MAX_TOKENS_PER_CONSTRAINT']
        converted_nodes = {}
        for node in nodes:
            # reduce the number of constraints
            node['constraints'] = self.__reduce_constraints(node['constraints'])
            converted_constraints = []
            if node['constraints'] != ['']:
                # Remove "junk symbols"
                node['constraints'] = self.__prettify_constraints(node['constraints'])
                # --------------------- TAL'S CODE START---------------------#
                # deduct constraints
                # node['constraints'] = self.__deduct_constraints(node['constraints'], 5)
                # --------------------- TAL'S CODE END---------------------#

                # Perform per-constraint styling on each node
                filtered_constraint_asts = self.__process_constraints_to_asts(node['constraints'])
                # Perform node-wide deduplication
                filtered_constraint_asts = self.__deduplicate_constraints(filtered_constraint_asts)
                # Convert to the nero format
                for constraint_ast in filtered_constraint_asts:
                    converted_constraints += constraint_ast.convert_list_to_nero_format()
            
            if not converted_constraints:
                converted_nodes[node['block_addr']] = []
            else:
                converted_nodes[node['block_addr']] = converted_constraints

        return converted_nodes

    def convert_json(self, filename: str):
        filesize = os.path.getsize(filename)
        if filesize == 0 or filesize > SYM_EXE_MAX_OUTPUT_TO_PROCESS:
            # print(f'Warning! file {filename} is empty or larger than {SYM_EXE_MAX_OUTPUT_TO_PROCESS}. Skipping.')
            # raise Exception #This is necessary as that calling function will omit this
            return False, None
        
        # _, result = self.get_stats_json(filename)
        # if result["precent_block_constraints"] < 0.3: # Check if there too few blocks with constraints. In the nero DS this should remove about 15% of the functions
        #     return False, None
        
        with open(filename, 'r') as function_file:
            initial_data = json.load(function_file)

        # convert operation - according to the Nero format
        exe_name = filename.split(os.sep)[-2]
        package_name = 'unknown'
        function_name = filename.split(os.sep)[-1][:-5]
        exe_name_split = list(filter(None, exe_name.split('_')))
        if len(exe_name_split) > 1:
            exe_name = exe_name_split[-1]
            package_name = exe_name_split[-2]
        
        # print(package_name, exe_name, function_name)

        #converted_data = {'func_name': OUR_API_TYPE + function_name, 'GNN_data': {}, 'exe_name': exe_name, 'package': package_name}
        converted_data = {'func_name': function_name, 'GNN_data': {}, 'exe_name': exe_name, 'package': package_name}
        # try:
        converted_data['GNN_data']['edges'] = self.__convert_edges(initial_data['GNN_DATA']['edges'])
        converted_data['GNN_data']['nodes'] = self.__convert_nodes(initial_data['GNN_DATA']['nodes'])
        #except Exception as e:
        #    print("file", filename)
        #    exit(1)
        
        #if self.portion_nodes_has_constraints(converted_data['GNN_data']['nodes']) < 0.25 or self.nodes_total_num_constraints(converted_data['GNN_data']['nodes']) < 20:
            #return False, None
        
        if self.nodes_total_num_constraints(converted_data['GNN_data']['nodes']) < 5:
            return False, None
        
        #if self.portion_nodes_has_constraints(converted_data['GNN_data']['nodes']) < 0.15:
            #return False, None
            
        #converted_data['GNN_data']['edges'], converted_data['GNN_data']['nodes'] = self.reduce_graph(converted_data['GNN_data']['edges'], converted_data['GNN_data']['nodes'])
        converted_filename = CONVERTED_DS_PREFIX + filename
        os.makedirs(os.path.dirname(converted_filename), exist_ok=True)
        with open(converted_filename, 'w') as function_file:
            jp_obj = str(encode(converted_data))
            function_file.write(jp_obj)
            
        return True, converted_filename

    '''
    Every two adjacent unconstrained blocks will be merged.
    '''
    def reduce_graph(self, edges, nodes):
        stop_loop = False
        while not stop_loop:
            stop_loop = True
            for e in edges:
                len_e = len(edges)
                len_n = len(nodes)
                src, dst = e
                if src != dst and len(nodes[src]) == 0 and len(nodes[dst]) == 0: # src and dst are unconstrained blockes, they should be merged
                    edges, nodes = self.merge_nodes(edges, nodes, src, dst)
                    stop_loop = False  # we updated the edges, the loop should start again (we can not update a list while iterating on it)
                    len_e_after_merge = len(edges)
                    len_n_after_merge = len(nodes)
                    assert len_e_after_merge == len_e - 1
                    assert len_n_after_merge == len_n - 1
                    break
        return edges, nodes
            
    '''
    Merge mode2 into node1
    '''
    def merge_nodes(self, edges, nodes, node1, node2):
        edges.remove((node1, node2))
        for i in range(len(edges)):
            if edges[i][0] == node2:
                edges[i] = (node1, edges[i][1])
            if edges[i][1] == node2:
                edges[i] = (edges[i][0], node1)
        del nodes[node2]
        return edges, nodes

    
    def portion_nodes_has_constraints(self, nodes):
        num_nodes_with_constraints = 0
        for _, constraints in nodes.items():
            if len(constraints) > 0:
                num_nodes_with_constraints = num_nodes_with_constraints + 1
        return num_nodes_with_constraints/len(nodes)
    
    def nodes_total_num_constraints(self, nodes):
        total_num_constraints = 0
        for _, constraints in nodes.items():
            total_num_constraints = total_num_constraints + len(constraints)
        return total_num_constraints
    
class OrganizeOutput:
    def __init__(self, dataset_name, file_locations, train_percentage, test_percentage, validate_percentage):
        self.dataset_name = dataset_name
        self.train_percentage = train_percentage
        self.validate_percentage = validate_percentage
        self.test_percentage = test_percentage
        self.file_locations = file_locations

    def print_information_and_fix(self):
        if self.train_percentage + self.test_percentage + self.validate_percentage != 100:
            print('CRITICAL! : all percentages don\'t add to 100')
        if self.train_percentage < self.validate_percentage + self.test_percentage:
            print('Warning! : not enough training')
        # TODO: add more warning and errors if needed

        self.test_percentage /= 100
        self.train_percentage /= 100
        self.validate_percentage /= 100

    def collect_files(self):
        """
        Aggregate all training, testing and validation files into single files.
        """
        #train_length = int(len(self.file_locations) * self.train_percentage)
        #test_length = int(len(self.file_locations) * self.test_percentage)
        #validate_length = len(self.file_locations) - train_length - test_length

        #print('num of train files: {}'.format(train_length))
        #print('num of test files: {}'.format(test_length))
        #print('num of validate files: {}'.format(validate_length))

        #random.shuffle(self.file_locations)

        #training_files = self.file_locations[:train_length]
        #testing_files = self.file_locations[train_length:train_length + test_length]
        #validating_files = self.file_locations[train_length + test_length:]

        ready_dir = 'ready_' + self.dataset_name

        if not os.path.exists(os.path.join('../ready_data', ready_dir)):
            os.mkdir(os.path.join('../ready_data', ready_dir))
        
        # --------------------- TAL'S CODE START---------------------#
        collect_to_file(self.file_locations, os.path.join(ready_dir, 'train.json'))
        collect_to_file(self.file_locations, os.path.join(ready_dir, 'test.json'))
        collect_to_file(self.file_locations, os.path.join(ready_dir, 'validation.json'))
        # --------------------- TAL'S CODE END---------------------#
        
        #collect_to_file(training_files, os.path.join(ready_dir, 'train.json'))
        #collect_to_file(testing_files, os.path.join(ready_dir, 'test.json'))
        #collect_to_file(validating_files, os.path.join(ready_dir, 'validation.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='enter dataset directory name (the one that is in preprocessed_data')
    parser.add_argument('--train', type=int, default=70, help='percent of functions in the train file')
    parser.add_argument('--test', type=int, default=20, help='percent of functions in the test file')
    parser.add_argument('--val', type=int, default=10, help='percent of functions in the validate file')
    parser.add_argument('--sample_path', type=int, default=1, help='select sample_path paths from the constraints of each block')
    parser.add_argument('--sample_constraint', type=int, default=10, help='select sample_constraint constraints from each path constraints of each block')
    parser.add_argument('--only_collect', dest='only_collect', action='store_true')
    parser.add_argument('--only_style', dest='only_style', action='store_true')
    args = parser.parse_args()

    print(f"Convert with sample_path={args.sample_path}, sample_constraint={args.sample_constraint}.")
    out_convertor = OutputConvertor(args.dataset_name, args.sample_path, args.sample_constraint)
    os.chdir('preprocessed_data')
    if not args.only_collect:
        out_convertor.clear_converted_dataset()
        out_convertor.load_all_files()
        out_convertor.convert_dataset()
    else:
        out_convertor.load_all_files()
        out_convertor.get_stats()
        
    collector = OrganizeOutput(args.dataset_name, out_convertor.converted_filenames, args.train, args.test, args.val)
    collector.print_information_and_fix()
    buff = input('collect converted files into train/val/test? [y/n]\n')
    if 'y' in buff or 'Y' in buff:
        collector.collect_files()


if __name__ == '__main__':
    main()
