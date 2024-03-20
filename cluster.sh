#!/bin/bash
#SBATCH --job-name=inbd
#SBATCH --ntasks=8
#SBATCH --mem=20
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy

# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)


#SBATCH --partition=normal
#SBATCH --qos=gpu

source /etc/profile.d/modules.sh
source /clusteruy/home/henry.marichal/miniconda3/etc/profile.d/conda.sh
conda activate inbd_gpu

# -------------------------------------------------------
#disco local SSD local al nodo. /clusteruy/home/henry.marichal se accede via NFS (puede ser realmente lento)
#el espacio local a utilizar se reserva dcon --tmp=XXXGb
LOCAL_NODE_DIR=/scratch/henry.marichal/
#HOME_DATASET_DIR=~/repos/INBD/dataset/EH
HOME_DATASET_DIR=~/datasets/inbd_4_2
#HOME_SEGMENTATION_MODEL=~/resultados/inbd_pinus_taeda_20240203_011323/model/2024-02-03_01h13m37s_segmentation_300e_x4_/model.pt.zip
#HOME_SEGMENTATION_MODEL=~/resultados/inbd_pinus_taeda_20240205_192015/model/2024-02-05_19h20m17s_segmentation_300e_x4_/model.pt.zip
HOME_SEGMENTATION_MODEL=~/repos/INBD/model/inbd_4/model.pt.zip
HOME_RESULTADOS_DIR=~/resultados/inbd_pinus_taeda_$(date +'%Y%m%d_%H%M%S')
HOME_RESULTADOS_MODEL_DIR=$HOME_RESULTADOS_DIR/model
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------


#other variables
#NODE_RESULTADOS_DIR=$LOCAL_NODE_DIR/inbd/resultados
#NODE_DATASET_DIR=$LOCAL_NODE_DIR/inbd/EH

#NODE_MODEL_RESULTADOS_DIR=$NODE_RESULTADOS_DIR/model
stdout_file="$HOME_RESULTADOS_DIR/stdout.txt"
stderr_file="$HOME_RESULTADOS_DIR/stderr.txt"
# Define a function to check the result of a command
check_command_result() {
    # Run the command passed as an argument
    "$@"

    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Command was successful."
    else
        echo "Command failed with an error."
        exit 1
    fi
}

####Prepare directories
#rm -rf $NODE_DATASET_DIR
#rm -rf $NODE_RESULTADOS_DIR
rm -rf $HOME_RESULTADOS_DIR

#check_command_result mkdir -p $NODE_DATASET_DIR
#check_command_result mkdir -p $NODE_RESULTADOS_DIR
check_command_result mkdir -p $HOME_RESULTADOS_DIR

####Move dataset to node local disk
#check_command_result cp  -r $HOME_DATASET_DIR $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program

cd ~/repos/INBD/
#python main.py train segmentation $HOME_DATASET_DIR/train_inputimages.txt $HOME_DATASET_DIR/train_annotations.txt --output $HOME_RESULTADOS_MODEL_DIR --epochs 300 > "$stdout_file" 2> "$stderr_file"
#next, train the inbd network
python main.py train INBD  $HOME_DATASET_DIR/train_inputimages.txt   $HOME_DATASET_DIR/train_annotations.txt   --segmentationmodel=$HOME_SEGMENTATION_MODEL  --output $HOME_RESULTADOS_MODEL_DIR > "$stdout_file" 2> "$stderr_file" #adjust path

# -------------------------------------------------------
#copy results to HOME
mkdir -p $HOME_RESULTADOS_DIR
#cp -r $NODE_RESULTADOS_DIR/* $HOME_RESULTADOS_DIR
#cp -r $NODE_DATASET_DIR/* $HOME_RESULTADOS_DIR
#delete temporal files
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $NODE_DATASET_DIR