#!/bin/bash
echo "THIS SCRIPT IS SUPPOSED TO WORK ON THE LAMBDA SERVERS ONLY."
echo "Use it anywhere else at your own risk!"

srun --gres=gpu:1 --pty /bin/bash
conda create -n tf1 python=3.7
conda activate tf1
pip install tensorflow-gpu==1.15

mkdir ~/symlinks
ln -s /usr/local/cuda/lib64/libcudart.so.10.2 ~/symlinks/libcudart.so.10.0  
ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10 ~/symlinks/libcublas.so.10.0  
ln -s /usr/local/cuda/lib64/libcufft.so.10.2 ~/symlinks/libcufft.so.10.0  
ln -s /usr/local/cuda/lib64/libcurand.so.10.2 ~/symlinks/libcurand.so.10.0  
ln -s /usr/local/cuda/lib64/libcusolver.so.10.2 ~/symlinks/libcusolver.so.10.0  
ln -s /usr/local/cuda/lib64/libcuparse.so.10.2 ~/symlinks/libcuparse.so.10.0

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/symlinks/" > ~/.bashrc
source ~/.bashrc

pip install -r requirements.txt