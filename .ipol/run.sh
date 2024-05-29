#!/bin/bash
set -e

# Read input parameters
input_0=$1
input_1=$2
output_0=$3
BIN=$4
HOME=$5

# Extract center from mask
python $BIN/.ipol/process_center.py --input_poly $input_1 --input_img $input_0 --output_img $output_0

#if [ -s inpainting_data_0.txt ]; then
  # File is not empty
#  python $BIN/.ipol/process_center.py --input_poly inpainting_data_0.txt --input_img $input_0 --output_img $output_0
  #rm inpainting_data_0.txt
#fi
#else
#  # File is  empty
#  $ANT_CENTER_DETECTOR/build/AntColonyPith --animated=false --input $input_0
#  stdout=$(python $BIN/.ipol/process_center.py --input $input_0 --type 1)
#
#fi
#Cy=$(echo $stdout | awk '{print $1}')
#Cx=$(echo $stdout | awk '{print $2}')

# Execute algorithm
#python $BIN/metric_influence_area.py --dt_filename $input_2 --gt_filename $input_1 --img_filename $input_0  --cx $Cx --cy $Cy --th $threshold --output_dir $HOME

#python $BIN/main.py --detection_path $input_0 --disk_name $disk_name --th $threshold --output_dir $HOME

#python $BIN/main.py --input $input --cx $Cx --cy $Cy --root $BIN --output_dir ./  --th_high $th_high --th_low $th_low --hsize $hsize --wsize $wsize --sigma $sigma --save_imgs 1
python $BIN/main.py inference $BIN/models/model.pt.zip $input_0 $output_0 --output $HOME/output/

