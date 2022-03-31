#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J active-ml-project-2
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need ... GB of memory per core/slot -- 
#BSUB -R "rusage[mem=3GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
##BSUB -M 18GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
### -- send notification at start -- 
###BSUB -B 
### -- send notification at completion -- 
####BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/Output_%J.out 
#BSUB -e logs/Error_%J.err 

cd ~/active-ml-project-2
source $HOME/.bashrc
conda activate active-ml-project-2
python testing_loop.py $args
