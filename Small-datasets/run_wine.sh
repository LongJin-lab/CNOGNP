#!/bin/bash


iterations=100   #200
particles=20
hidden_dim=10
w=1
eta=0.1




# CNO
# device='0'
# algorithm='CNO'
# loc='CNO'  #
# c1=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.87 0.03 0.55 0.19 0.72 0.40 0.91 0.28 0.66 0.05)
# c2=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.14 0.95 0.38 0.61 0.02 0.89 0.47 0.70 0.16 0.53)
# row_num=1
# for ((i=0;i<1;i++));do #c
#     for ((j=0;j<10;j++));do # number of repeated experiments
#         CUDA_VISIBLE_DEVICES=${device} nohup /home/ps/anaconda3/envs/sulw/bin/python wine.py  --algorithm ${algorithm} --max_iterations ${iterations} --n_particles ${particles} --hidden_dim ${hidden_dim} --eta ${eta} --omega ${w} --c1 ${c1[i]} --c2 ${c2[i]} --loc ${loc} --row_num $((i + row_num)) --column_num ${j} > ./log/wine/${loc}/CNO-I_${iterations}-P_${particles}-eta_${eta}-[w,c1,c2]_[${w},${c1[i]},${c2[i]}]-${j}.txt 2>&1
#     done;
# done;



# CNOGNP
# device='0'
# algorithm='CNOGNP'
# # lambda_reg=0.0005
# c1=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.87 0.03 0.55 0.19 0.72 0.40 0.91 0.28 0.66 0.05)
# c2=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.14 0.95 0.38 0.61 0.02 0.89 0.47 0.70 0.16 0.53)
# loc='CNOGNP'  #
# # row_num=111    #1->0.5 23->0.1 45->0.01 67->0.001 89->0.0001 111->0.0005
# lambda_reg=(0.5 0.1 0.01 0.01 0.001 0.0001 0.0005)
# row_num=(1 23 45 45 67 89 111)

# for ((k=0;k<7;k++));do #c
#     for ((i=0;i<20;i++));do #c
#         for ((j=0;j<10;j++));do # number of repeated experiments
#             CUDA_VISIBLE_DEVICES=${device} nohup /home/ps/anaconda3/envs/sulw/bin/python wine.py  --algorithm ${algorithm} --max_iterations ${iterations} --n_particles ${particles} --hidden_dim ${hidden_dim} --lambda_reg ${lambda_reg[k]} --eta ${eta} --omega ${w} --c1 ${c1[i]} --c2 ${c2[i]} --loc ${loc} --row_num $((i + row_num[k])) --column_num ${j} > ./log/wine/${loc}/CNOGNP_${lambda_reg[k]}-I_${iterations}-P_${particles}-eta_${eta}-[w,c1,c2]_[${w},${c1[i]},${c2[i]}]-${j}.txt 2>&1
#         done;
#     done;
# done;

# PSO
# device='0'
# algorithm='PSO'
# c1=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.87 0.03 0.55 0.19 0.72 0.40 0.91 0.28 0.66 0.05)
# c2=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.14 0.95 0.38 0.61 0.02 0.89 0.47 0.70 0.16 0.53)
# loc='PSO'  #
# row_num=1    #
# for ((i=0;i<20;i++));do #c
#     for ((j=0;j<10;j++));do # number of repeated experiments
#         CUDA_VISIBLE_DEVICES=${device} nohup /home/ps/anaconda3/envs/sulw/bin/python wine.py  --algorithm ${algorithm} --max_iterations ${iterations} --n_particles ${particles} --hidden_dim ${hidden_dim}  --omega ${w} --c1 ${c1[i]} --c2 ${c2[i]} --loc ${loc} --row_num $((i + row_num)) --column_num ${j} > ./log/wine/${loc}/PSO-I_${iterations}-P_${particles}-[w,c1,c2]_[${w},${c1[i]},${c2[i]}]-${j}.txt 2>&1
#     done;
# done;

# EGD 
# device='1'
# algorithm='CNO'
# particles=1
# c1=(0)
# c2=(0)
# loc='EGD'  #
# row_num=1    #
# for ((i=0;i<1;i++));do #c
#     for ((j=0;j<10;j++));do # number of repeated experiments
#         CUDA_VISIBLE_DEVICES=${device} nohup /home/ps/anaconda3/envs/sulw/bin/python wine.py  --algorithm ${algorithm} --max_iterations ${iterations} --n_particles ${particles} --hidden_dim ${hidden_dim}  --omega ${w} --c1 ${c1[i]} --c2 ${c2[i]} --loc ${loc} --row_num $((i + row_num)) --column_num ${j} > ./log/wine/${loc}/EGD-I_${iterations}-P_${particles}-[w,c1,c2]_[${w},${c1[i]},${c2[i]}]-${j}.txt 2>&1
#     done;
# done;

# PSO-BFGS
# device='3'
# algorithm='PSO-BFGS'
# loc='PSO-BFGS'  #
# c1=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.87 0.03 0.55 0.19 0.72 0.40 0.91 0.28 0.66 0.05)
# c2=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.14 0.95 0.38 0.61 0.02 0.89 0.47 0.70 0.16 0.53)
# row_num=1
# for ((i=15;i<20;i++));do #c
#     for ((j=0;j<10;j++));do # number of repeated experiments
#         CUDA_VISIBLE_DEVICES=${device} nohup /home/ps/anaconda3/envs/sulw/bin/python wine.py  --algorithm ${algorithm} --max_iterations ${iterations} --n_particles ${particles} --hidden_dim ${hidden_dim} --eta ${eta} --omega ${w} --c1 ${c1[i]} --c2 ${c2[i]} --loc ${loc} --row_num $((i + row_num)) --column_num ${j} > ./log/wine/${loc}/PSO-BFGS-I_${iterations}-P_${particles}-eta_${eta}-[w,c1,c2]_[${w},${c1[i]},${c2[i]}]-${j}.txt 2>&1
#     done;
# done;


