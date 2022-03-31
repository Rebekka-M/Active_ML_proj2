#!/bin/bash

#counter=0
#for sigma in 0.1; do
#for n in 10; do
#for s in 1; do
#counter=$((counter+1))
#bsub -env args="seed=$s|data_dim=1000|cluster_std=$sigma|n_train=$n|test=False|experiment=($counter)" < gplvm.sh;
#done
#done
#done
#echo "$counter"

rm -rf ./logs/*
#counter=1s
for s in {1..20}; do
bsub -env args="$s" < submit.sh;
counter=$((counter+1))
done
echo "$counter"
