#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J gplvm
### -- ask for number of cores (default: 1) -- 
#BSUB -n 2
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need ... GB of memory per core/slot -- 
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
##BSUB -M 17GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
###BSUB -u s204113@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/Output_%J.out 
#BSUB -e logs/Error_%J.err 

cd /zhome/30/9/156503/code/gp-lvm-cluster-analysis/
export PATH=/zhome/30/9/156503/anaconda3/bin:$PATH
source activate gplvm
python3 -c "from main import *; run()" $args
