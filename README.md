# Binary2Name
## Automatic Detection for Binary Code Functionality

This project was developed by [Ittay Alfassi](https://github.com/ittay-alfassi) and [Itamar Juwiler](https://github.com/itamar1208) as a part of the [Project in Computer Security](https://webcourse.cs.technion.ac.il/236349/Spring2021/) course at Technion - Israel Institute of Technology. 
Project Advisors: Dr. Gabi Nakibly and Dr. Yaniv David. 

## Introduction:
The main motivation for this project is to be a helpful tool for researchers of binary code.
We started with binary datasets as input and used [angr](https://angr.io), a symbolic analysis tool to get an intermediate representation of the code.
From there, came the most extensive step in the project which was to preprocess the intermediate code in preparation to be used as input to a neural network. We used a deep neural network adopted from [Nero](https://github.com/tech-srl/nero), which is intended for the same problem but used a different approach.

We suggest reading our report about this project (final_report.pdf) before running the code.

Getting started:
=====================
## Requirements:
    -   Python 3.6 and up
    -   all packages shown in requirements.txt 

### Setting up TensorFlow 1.15 with GPU
In order to install TensorFlow 1.15 with GPU support, first install Miniconda.
The setup_venv.sh script should perform all the following steps automatically :)

Follow these steps:

To install the package:
> srun --gres=gpu:2 --pty /bin/bash   #This is needed to get shell on one of worker servers
> conda create -n tf1 python=3.7
> conda activate tf1
> pip install tensorflow-gpu==1.15

Now, several symlinks are required to connect to the server's CUDA.
You need to have the following symlinks, saved in a home directory of your choosing:
> libcublas.so.10.0 -> /usr/lib/x86_64-linux-gnu/libcublas.so.10
> libcudart.so.10.0 -> /usr/local/cuda/lib64/libcudart.so.10.2
> libcufft.so.10.0 -> /usr/local/cuda/lib64/libcufft.so.10
> libcurand.so.10.0 -> /usr/local/cuda/lib64/libcurand.so.10
> libcusolver.so.10.0 -> /usr/local/cuda/lib64/libcusolver.so.10
> libcusparse.so.10.0 -> /usr/local/cuda/lib64/libcusparse.so.10

These files will only exist on the nodes, to make sure to use `srun --pty /bin/bash` before you try to set the links.
You can make a link using `ln -s`. For example:
> ln -s /usr/local/cuda/lib64/libcudart.so.10.2 ~/libcudart.so.10.0  

(make sure to use the location of YOUR symlink in the last argument.)

Finally, to add these symlinks to the PATH, use the `export` command. For example:
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/

Note that this addition is not permanent.

## Full preprocessing and training:

The bash script `run_whole_pipeline.sh` runs the project pipeline as a whole, and should achieve the results presented in the report.

If you're running on the Lambda server, make sure to use srun!

### Extract our datasets: 
The bash script `extract_data.sh` extracts all the zipped data that we used and generated.

## Source code file description:
  * `paths_constraints_main.py` is the python script that performs the basic symbolic execution. It reads its datasets from the `our_dataset` directory and saves its results to the `preprocessed_data` directory.

  * `output_converter.py` is the file that applies constraint styling.  It reads its input from `preprocessed_data/<dir_name>`. The converted output will be saved under `preprocessed_data/Converted_<dir_name>`.
    The aggregated output will be saved under `ready_data/ready_<dir_name>`.

  * `nero/preprocess.py` is the file that processes the files from `ready_data` and prepares them for execution by Nero. It reads its input from the any specified file, but usually uses `nero/procedure_representations/raw`.

  * `nero/gnn.py` is the file that activates Nero's model. It reads its input from any specified file, but usually uses `nero/procedure_representations/preprocessed`.

For more information on the Nero source code, reading the README of [Nero](https://github.com/tech-srl/nero) is recommended.
