#!/bin/bash

# set the directory containing the files
directory="./env_files"

# loop over all files in the directory
for file in "$directory"/*
do
  # run the command with the file name as input
  conda env create -f "$file"
done