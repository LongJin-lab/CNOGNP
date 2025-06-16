#!/bin/bash


model=('r18' 'r34' 'r50')

lr=0.0001
epochs=10
c1=0.00001
c2=0.00001
initial_noise_level=0.0000001
lambda_gnp=0.1
loc='cnognp_finetune'  
device='0'

for ((h=1;h<2;h++));do #model
    for ((j=0;j<1;j++));do 
        CUDA_VISIBLE_DEVICES=${device} nohup /home/qinch21/.conda/envs/pt37_backup/bin/python finetune_cnognp.py --c1 ${c1} --c2 ${c2} --initial_noise_level ${initial_noise_level} --lambda_gnp ${lambda_gnp} --eta ${lr} --cnognp_epochs ${epochs} --arch ${model[h]} --load_path "/home/qinch21/lab415/qc/paper-CNOGNP/one-card/model_pth/${model[h]}/imagenet-${model[h]}.pth" --save_path "/home/qinch21/lab415/qc/paper-CNOGNP/one-card/log/${loc}/imagenet-${loc}-${model[h]}-epoch${epochs}-[c1,c2,in,lam]-[${c1},${c2},${initial_noise_level},${lambda_gnp}].pth"> ./log/${loc}/imagenet-finetune-cnognp-${model[h]}-epochs${epochs}-[c1,c2,in,lam]-[${c1},${c2},${initial_noise_level},${lambda_gnp}]_${j}.txt 2>&1
    done;
done;



