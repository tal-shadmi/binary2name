import angr
import argparse
from angrutils import *


def draw(filename, func_name):
    proj = angr.Project(filename, load_options={'auto_load_libs':False})
    func_addr = proj.loader.main_object.get_symbol(func_name)
    start_state = proj.factory.blank_state(addr=func_addr.rebased_addr)
    cfg = proj.analyses.CFGFast()
    plot_cfg(cfg, "func_cfg", asminst=True, remove_imports=True, remove_path_terminator=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_name", type=str, required=True)
    parser.add_argument("--func_name", type=str, required=True)
    args = parser.parse_args()
    draw(args.binary_name, args.func_name)