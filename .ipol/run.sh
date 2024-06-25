#!/bin/bash
set -e

# Read input parameters
input_0=$1
input_1=$2
output_0=$3
BIN=$4
HOME=$5
wsize=$6
hsize=$7

# Extract center from mask
python $BIN/.ipol/preprocessing.py --input_poly $input_1 --input_img $input_0 --mask_path $output_0 --hsize $hsize --wsize $wsize

python $BIN/main.py inference $BIN/models/model.pt.zip $input_0 $output_0 --output $HOME/output/

