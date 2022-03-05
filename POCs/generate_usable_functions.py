import os
import argparse
from typing import List, Dict, Set
from angr import Project

DATASETS_ROOT = "our_dataset"


def is_ELF(file_path: str) -> bool:
    with open(file_path, "rb") as f:
        magic = f.read(4)
    return magic == b"\x7fELF"


def get_func_names(binary_name: str, analyzed_functions: Set[str] = set()) -> List[str]:
    """
    get functions that are suitable for analysis, (funcs that are defined in the binary and not libc funcs...)
    """
    proj = Project(binary_name, auto_load_libs=False)
    _ = proj.analyses.CFGFast()
    return list(filter(None, [f.name if f.binary_name == binary_name and (not f.is_plt) and not f.name.startswith(
        "sub_") and not f.name.startswith("_") and f.name not in analyzed_functions
        else None for f in proj.kb.functions.values()]))


def generate_names_file(bin_base_dir: str, bin_name: str, names_file_dir: str, suffix:str ="_names.txt") -> str:
    """
    Generate a file containing the function names of a single binary.

        :param bin_base_dir: base directory of the binary
        :type bin_base_dir: str
        :param bin_name: the binary file's name
        :type bin_name: str
        :param names_file_dir: base directory for the names file
        :type names_file_dir: str
        :param suffix: the suffix added to the binary's names file, defaults to "_names.txt"
        :type suffix: str, optional
        :return: the path of binary's name file
        :rtype: str
    """
    # Verify this is a legitimate file
    bin_file_path = os.path.join(bin_base_dir, bin_name)
    if not os.path.isfile(bin_file_path) or not is_ELF(bin_file_path):
        raise RuntimeError(f"The {bin_file_path} ELF file was not found.")

    # Run a bash script for function name extractiopn
    names_file_path = os.path.join(names_file_dir, bin_name + suffix)
    with open(names_file_path, "w") as f:
        f.writelines(get_func_names(bin_file_path))
    # os.system(  f"nm {bin_file_path} --demangle 2> {names_file_path} | " +  # Read the symbols from the file
    #             f"grep ' T ' | " +                                          # Keep only symbols from the text section
    #             f"cut -d ' ' -f 3" +                                        # Keep only the function name
    #             f" > {names_file_path}")                                    # save to the names file
    
    return names_file_path


def remove_names_files(names_file_dir: str, suffix: str="_names.txt"):
    """
    Remove all file names generated in the process.

        :param names_file_dir: base directory for the names file
        :type names_file_dir: str
        :param suffix: the suffix added to the binary's names file, defaults to "_names.txt"
        :type suffix: str, optional
    """
    for filename in os.listdir(names_file_dir):
        if suffix in filename:
            os.remove(os.path.join(names_file_dir, filename))
    

def generate_names_list(names_files_list: List[str]) -> List[str]:
    """
    Generate a list of all names in the dataset.
    """
    names_list: List[str] = []

    for names_file in names_files_list:
        with open(names_file, "r") as f:
            names = [l.strip() for l in f.readlines()] # Read the names from each file

        if names == [] or "File format not recognized" in names[0]: # Make sure the file is not from a failed analysis attempt
            continue
        names_list += names

    return names_list


def generate_token_histogram(names_list: List[str], delimiter: str="_") -> Dict[str, int]:
    """
    Generate a histogram of all function-name tokens.

        :param names_list: list of all function names in the dataset
        :type names_list: List[str]
        :param delimiter: delimiters seperating tokens in the dataset's function names, defaults to "_"
        :type delimiter: str, optional
        :return: a dictionary from each token to it's occurence count
        :rtype: Dict[str, int]
    """
    token_hist: Dict[str, int] = {}
    for name in names_list:
        for token in name.split(delimiter):
            if token in token_hist:
                token_hist[token] += 1
            else:
                token_hist[token] = 1
    return token_hist


def generate_usable_funcs_names_list(names_list: List[str], token_hist: Dict[str, int],
                                threshold: int, delimiter: str="_") -> List[str]:
    """
    Generate a list of all the usable function names in the dataset.
    A function is considered usable if all of its tokens appear at least k times in the database.

        :param names_list: list of all function names in the dataset
        :type names_list: List[str]
        :param token_hist: a dictionary from each token to it's occurence count
        :type token_hist: Dict[str, int]
        :param threshold: minimal k value
        :type threshold: int
        :param delimiter: delimiters seperating tokens in the dataset's function names, defaults to "_"
        :type delimiter: str, optional
        :return: list of all the usable function names in the dataset
        :rtype: List[str]
    """
    usable_functions: List[str] = []
    for name in names_list:
        tokens = name.split(delimiter)
        if all([token in token_hist and token_hist[token] >= threshold for token in tokens]) \
            and name not in usable_functions:
            usable_functions.append(name)
                
    usable_functions.sort() # Sort for compatibility with old format
    return usable_functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--namefiles_dir", type=str, default=".")
    parser.add_argument("--occurence_threshold", type=int, default=5)
    parser.add_argument("--functions_filename", type=str, default="usable_functions_names.txt")
    parser.add_argument("--keep_namefiles", dest="keep_namefiles", action="store_true")
    args = parser.parse_args()

    dataset_dir = os.path.join(DATASETS_ROOT, args.dataset_dir)
    namefiles_dir = os.path.join(DATASETS_ROOT, args.namefiles_dir)

    if not os.path.isdir(namefiles_dir):
        os.mkdir(namefiles_dir)

    try:
        bin_filenames_in_dir = os.listdir(dataset_dir) # Get names of binary files
        names_files_list: List[str] = []
        for bin_filename in bin_filenames_in_dir:
            if is_ELF(os.path.join(dataset_dir, bin_filename)): # Don't add names file to analysis
                names_files_list.append(generate_names_file(dataset_dir, bin_filename, namefiles_dir))
        
    except FileNotFoundError as e:
        print(f"base dir {dataset_dir} not found. Exiting")
        print(e)
        exit(0)

    except RuntimeError as e:
        print("Runtime Error Occured. Exiting")
        print(e)
        exit(0)

    # Begin processing pipeline
    names_list = generate_names_list(names_files_list)
    token_hist = generate_token_histogram(names_list)
    usable_functions = generate_usable_funcs_names_list(names_list, token_hist, args.occurence_threshold)
    with open(os.path.join(namefiles_dir, args.functions_filename), "w") as f:
        for func in usable_functions:
            f.write(func + "\n")

    if not args.keep_namefiles:
        remove_names_files(namefiles_dir)


if __name__ == "__main__":
    main()


# python3 generate_usable_functions.py --dataset_dir coreutils_ds --namefiles_dir coreutils_namefiles --keep_namefiles
