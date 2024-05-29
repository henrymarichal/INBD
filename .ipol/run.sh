#!/bin/bash
set -e

# Read input parameters
input_0=$1
BIN=$2
HOME=$3

# Extract center from mask
python $BIN/.ipol/process_center.py --input inpainting_data_0.txt

if [ -s inpainting_data_0.txt ]; then
  # File is not empty
  stdout=$(python $BIN/.ipol/process_center.py --input inpainting_data_0.txt --type 0)
  rm inpainting_data_0.txt
fi
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
#python main.py inference models/model.pt.zip assets/F02c.png assets/F02c_mask.png --output $HOME/output/

