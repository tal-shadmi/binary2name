#!/bin/bash

echo "Welcome! This is an execution bash script for the coreutils results, using only 65% of the data (assuming symbolic execution is already done)"
echo "Remember - If you're running on Lambda, usr srun!"
echo "First run 'srun -c 'number of cpus you would like to use' --gres=gpu:'number of gpus you would like to use' --pty \bin\bash' to get a hold of one of the lambda nodes"
echo "Also remember to have all your modules installed - angr and TensorFlow 1 included."

echo "Running output processing and conversion"
yes | python3 output_converter.py --dataset_name coreutils

echo "Removing empty lines (feel free to find a better way)"
grep . ready_data/ready_coreutils/train.json > ready_data/ready_coreutils/train.json_
mv ready_data/ready_coreutils/train.json_ ready_data/ready_coreutils/train.json
grep . ready_data/ready_coreutils/validation.json > ready_data/ready_coreutils/validation.json_
mv ready_data/ready_coreutils/validation.json_ ready_data/ready_coreutils/validation.json
grep . ready_data/ready_coreutils/test.json > ready_data/ready_coreutils/test.json_
mv ready_data/ready_coreutils/test.json_ ready_data/ready_coreutils/test.json

echo "Copying to nero's internal data directory"
mkdir -p nero/coreutils/procedure_representations/raw/bin2name
cp ready_data/ready_coreutils/train.json nero/coreutils/procedure_representations/raw/bin2name/train.json
cp ready_data/ready_coreutils/validation.json nero/coreutils/procedure_representations/raw/bin2name/validation.json
cp ready_data/ready_coreutils/test.json nero/coreutils/procedure_representations/raw/bin2name/test.json

echo "Running Nero's preprocessing"
cd nero/coreutils
python3 ../preprocess.py -trd procedure_representations/raw/bin2name/train.json -ted procedure_representations/raw/bin2name/test.json -vd procedure_representations/raw/bin2name/validation.json -o data

echo "Copying to nero's internal preprocessed directory"
mkdir -p procedure_representations/preprocessed/bin2name
mv data.dict procedure_representations/preprocessed/bin2name
mv data.train procedure_representations/preprocessed/bin2name
mv data.val procedure_representations/preprocessed/bin2name
mv data.test procedure_representations/preprocessed/bin2name

echo "Running Nero"
python3 -u ../gnn.py  --data procedure_representations/preprocessed/bin2name/data --test procedure_representations/preprocessed/bin2name/data.test --save new_model/model --gnn_layers 4 > coreutils_out_log.txt 2> coreutils_error_log.txt

echo "That's All, Folks!"