#!/bin/bash -l
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/vgg16_train.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/vgg16_train.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Network_Trim_VGG_Full
#SBATCH -N 1
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
#python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd Network_Trimming_Pytorch

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# run code here. full train.
echo "Starting full training for VGG16"
python3 -m train_vgg_cifar10 --epoch 50

echo "Prune VGG16"
 python3 -m trim_vgg_cifar10

# echo "Fine tune VGG16"


deactivate