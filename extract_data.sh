#!/bin/bash

echo "Extracting binary datasets"
tar xzf our_dataset/coreutils_ds.tar.gz -C our_dataset
tar xzf our_dataset/nero_ds.tar.gz -C our_dataset

#echo "Extracting symbolic analysis output - might take a while"
#tar xzf preprocessed_data/coreutils.tar.gz -C preprocessed_data
#tar xzf preprocessed_data/nero.tar.gz -C preprocessed_data

#echo "Extracting processed and converted analysis output"
#tar xzf preprocessed_data/Converted_coreutils.tar.gz -C preprocessed_data
#tar xzf preprocessed_data/Converted_nero.tar.gz -C preprocessed_data

#echo "Extracting pre-nero ready data (partitioned to train, val, test)"
#tar xzf ready_data/ready_coreutils.tar.gz -C ready_data
#tar xzf ready_data/ready_nero.tar.gz -C ready_data

#echo "Extracting nero's ready data (internal procedure_representations directory)"
#tar xzf nero/procedure_representations.tar.gz -C nero

echo "Done extracting - enjoy :)"
