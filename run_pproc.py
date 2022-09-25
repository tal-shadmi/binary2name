import os
import sys
import argparse
import logging
import time
from datetime import datetime
from multiprocessing import Process
from select import select
from typing import Dict, Tuple
# from random import randint


# def run_single_bin_test(output_dir: str, dataset_dir: str, bin_timeout: str, bin_idx:int):
#     res = 0
#     for _ in range(randint(2**26, 2**27)):
#         res += 1

def run_single_bin(args: argparse.Namespace, bin_idx: int):
    """Run an instance of the preprocessing script on a single binary."""

    usables_flag = "--no_usables_file " if args.no_usables_file else ""
    out_log = os.path.join(args.log_dir, f"{bin_idx}_out.log")
    err_log = os.path.join(args.log_dir, f"{bin_idx}_err.log")
    cmd_line = f"timeout {args.bin_timeout} python3 paths_constraints_main.py --binary_idx={bin_idx} --output={args.output_dir} " + \
    f" --dataset={args.dataset_dir} --mem_limit={args.mem_limit} {usables_flag} 2> {err_log} > {out_log}"
    os.system(cmd_line)


def dispatch_process(args: argparse.Namespace, bin_idx:int, processes: Dict[int, Tuple[int, Process]]) -> Dict[int, Tuple[int, Process]]:
    """Dispatch a process to analyze a single binary."""
    bin_idx = abs(bin_idx)
    print(f'dispatching idx {bin_idx} at {datetime.today()}')
    proc = Process(target=run_single_bin, args=(args, bin_idx))
    proc.start()
    processes[proc.sentinel] = (bin_idx, proc)
    return processes


def collect_process(processes: Dict[int, Tuple[int, Process]]):
    # Wait for tick from finished sentinel
    finished_procs, _, _ = select(processes.keys(), [], [])

    # Remove it from the processes and sentinels lists
    for sentinel in finished_procs:
        print(f'idx {processes[sentinel][0]} completed at {datetime.now()}')
        del processes[sentinel]


def run_preprocess(args: argparse.Namespace, base_dataset_dir:str = "our_dataset"):
    """Run the preprocessing on all files in the args.dataset_dir."""

    processes: Dict[int, Tuple[int, Process]] = {}

    # Calculate number of binaries
    full_dataset_dir = os.path.join(base_dataset_dir, args.dataset_dir)
    bin_count = len(os.listdir(full_dataset_dir))
    
    if args.reverse_order:
        start_of_bin = -1 * bin_count    # [-bin_count, -bin_count+1, -bin_count+2, ..., 0]
    else:
        start_of_bin = 0                 # [0, 1, 2, ..., bin_count]
    end_of_bin = start_of_bin + bin_count
    # Fill CPUs with jobs
    curr_bin = start_of_bin
    for _ in range(min(args.cpu_no, bin_count)):
        processes = dispatch_process(args, curr_bin, processes)
        curr_bin += 1

    while curr_bin < end_of_bin:
        collect_process(processes)

        # Build and run a new process
        try:
            processes = dispatch_process(args, curr_bin, processes)
            curr_bin += 1
        except OSError as e:
            if e.errno == 12: # Cannot allocate memory
                logging.error(f"run_pproc did not have enough memory to allocate idx {curr_bin}. Sleeping it off (10 minutes)...")
                time.sleep(60*10)
                logging.error(f"Woke up from nap - retrying to allocate new jobs.")
            else:
                raise e
                

    # Wait for all remaining processes
    while processes:
        collect_process(processes)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument('--cpu_no', type=int, default=40)
    parser.add_argument("--mem_limit", type=int, default=20)
    parser.add_argument('--bin_timeout', type=str, default='1000m')
    parser.add_argument("--no_usables_file", dest="no_usables_file", action="store_true")
    parser.add_argument("--reverse", dest="reverse_order", action="store_true") # process the files from the last to the first
    args = parser.parse_args()

    # Make a directory for log files
    os.makedirs(args.log_dir, exist_ok=True)
    sys.stdout = open(os.path.join(args.log_dir, "b2n_pproc_out.log"), "w")
    sys.stderr = open(os.path.join(args.log_dir, "b2n_pproc_err.log"), "w")

    logging.getLogger().setLevel('DEBUG')
    print('STARTING!!!!!!!')
    run_preprocess(args)
    print('DONEDONEDONE!!!!!!!')

    sys.stdout.close()
    sys.stderr.close()


if __name__ == "__main__":
    main()

# python3 run_pproc.py --output_dir nero_test --dataset_dir nero_ds/TRAIN --log_dir nero_logs --cpu_no 30 --mem_limit 45 --no_usables_file &
# srun -c 60 python3 run_pproc.py --output_dir coreutils_new --dataset_dir coreutils_ds --log_dir coreutils_logs --cpu_no 30 --mem_limit 45 --no_usables_file &
