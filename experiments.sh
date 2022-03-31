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
for sigma in 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 10.0 100.0 ; do
for n in {10..100..10}; do
for s in {1..10}; do
for pca in True False; do
bsub -env args="seed=$s|data_dim=1000|cluster_std=$sigma|n_train=$n|test=False|experiment=($counter)|gp_latent_init_pca=$pca" < gplvm.sh;
counter=$((counter+1))
done
done
done
done
echo "$counter"
