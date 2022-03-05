import re


def calc_token_number(constraint: str) -> int:
    dirty_tokens = re.split(',|\||\(|\)|<|>', constraint)
    clean_tokens = list(filter(None, dirty_tokens))
    return len(clean_tokens)


def get_instruction_count(block_dict: dict, delim="    ") -> int:
    assert "instructions" in block_dict
    return len(list(filter(None, block_dict["instructions"].split(delim))))


def get_constraint_count(block_dict: dict, delim="    ") -> int:
    assert "constraints" in block_dict
    constraints = block_dict["constraints"]

    if not constraints:
        return 0

    if isinstance(constraints[0], str):  # Support old version, in which every path is represented at as a string.
        constraints = [list(filter(None, con.split(delim))) for con in constraints]

    lengths = [len(con_list) for con_list in constraints]
    return sum(lengths)


def get_constraint_len(block_dict: dict, delim="   ") -> int:
    assert "constraints" in block_dict
    constraints = block_dict["constraints"]

    if not constraints:
        return 0

    if isinstance(constraints[0], str):  # Support old version, in which every path is represented at as a string.
        constraints = [list(filter(None, con.split(delim))) for con in constraints]

    lengths = [[calc_token_number(con) for con in con_list] for con_list in constraints]
    lengths = [sum(length) for length in lengths]
    return sum(lengths)


def calc_constraint_stats(file_dict: dict, filename: str) -> dict:
    # calculating the amount of constraints in each node and doing an average
    total_count = [get_constraint_count(node_dict) for node_dict in file_dict['nodes']]
    node_count = len(file_dict['nodes'])
    total_length = [get_constraint_len(node_dict) for node_dict in file_dict['nodes']]

    return {'filename': filename, 'total_node_num': node_count, 'constraint_amount_distrib': total_count,
            'constraint_total_count': sum(total_count), 'constraint_amount_avg': sum(total_count) / node_count,
            'constraint_len_total': sum(total_length),
            'constraint_len_distrib': total_length, 'constraint_len_avg': sum(total_length) / node_count}
