#!/bin/bash

model=('r18' 'r34' 'r50')

lr=0.0001
epochs=10
loc='sgd_finetune'  
device='3'
for ((h=1;h<2;h++));do #model
    for ((j=0;j<1;j++));do 
        CUDA_VISIBLE_DEVICES=${device} nohup /home/qinch21/.conda/envs/pt37_backup/bin/python finetune_sgd.py  --ft_lr ${lr} --ft_epochs ${epochs} --arch ${model[h]} --load_path "/home/qinch21/lab415/qc/paper-CNOGNP/one-card/model_pth/${model[h]}/imagenet-${model[h]}.pth" --save_path "/home/qinch21/lab415/qc/paper-CNOGNP/one-card/log/${loc}/imagenet-${model[h]}-epoch${epochs}.pth"> ./log/${loc}/imagenet-finetune-sgd-${model[h]}-epochs${epochs}_${j}.txt 2>&1
    done;
done;



