import argparse
import os
import json
from typing import List, Dict


def create_distribution_dict(functions_data: List, field: str) -> (Dict, int):
    data = {}
    total = 0
    for function in functions_data:
        for value in function[field]:
            if value not in data.keys():
                data[value] = 1
            else:
                data[value] += 1
            total += 1
    return data, total


def run(histogram_file, destination_file):

    with open(histogram_file, 'r') as file:
        data = json.load(file)

    amount_of_constraints, total_cons_amount = create_distribution_dict(data, 'constraint_amount_distrib')
    token_number_in_constraints, total_token_amount = create_distribution_dict(data, 'constraint_len_distrib')

    with open(destination_file, 'w') as file:
        json.dump(
            {
                'amount_of_constraints': amount_of_constraints,
                'token_number_in_constraints': token_number_in_constraints,
                'total_cons_amount': total_cons_amount,
                'total_token_amount': total_token_amount
            },
            file, indent=4, sort_keys=True
        )


def to_minmax(data: Dict) -> List[int]:
    to_return = []
    for key in data.keys():
        to_return.append(int(key))
    return to_return


def interact(file):
    data = json.load(file)
    buff_type = input('enter histogram to check\n0=constraints_amount\n1=token_numbers\n')
    while buff_type != '0' and buff_type != '1':
        buff_type = input('invalid choice, choose again\n0=constraints_amount\n1=token_numbers\n')
    if buff_type == '0':
        buff_type = 'amount_of_constraints'
        total = data['total_cons_amount']
    else:
        buff_type = 'token_number_in_constraints'
        total = data['total_token_amount']

    print('the maximum range values are min:{} and max:{}'.format(min(to_minmax(data[buff_type])), max(to_minmax(data[buff_type]))))
    buff = input('enter range of numbers to check\nsyntax: <start>-<stop>\nenter \'end\' to end\n')
    while buff != 'end':
        calc_sum = 0
        for i in range(int(buff[0]), int(buff[2])):
            if str(i) in data[buff_type].keys():
                calc_sum += data[buff_type][str(i)]
        percent = (calc_sum / total) * 100.0
        print('the percentage in the range you offered is {:.4f}%'.format(percent))
        buff = input('enter new to check\nsyntax: <start>-<stop>\nenter \'end\' to end\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist_dir', type=str, required=True)
    parser.add_argument('--update', type=int, required=True)
    args = parser.parse_args()
    histogram_file = os.path.join('preprocessed_histograms/' + args.hist_dir, 'per_file_hist.json')
    destination = os.path.join('preprocessed_histograms/' + args.hist_dir, 'distribution_dictionaries.json')
    if args.update == 1:
        run(histogram_file, destination)
    else:
        try:
            with open(destination, 'r') as file:
                interact(file)
        except FileNotFoundError:
            print('trying to interact with non-existent file, try updating first (run with --update=True)')


if __name__ == '__main__':
    main()
