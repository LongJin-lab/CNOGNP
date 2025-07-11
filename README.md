# CNOGNP
# Collaborative Neurodynamic Optimization with Gradient Norm Penalty for Generalization Improvement
This repository contains the official implementation for the paper: **"Collaborative Neurodynamic Optimization with Gradient Norm Penalty for Generalization Improvement"**.

## Abstract

Some machine learning tasks that emphasize the generalization property, e.g., a supervised classification task, can be transformed into nonconvex optimization problems. In general, when solving these nonconvex optimization problems, many optimization algorithms, including heuristic algorithms, aim to find the global minimum (with the smallest objective function value), which often results in limited generalization performances. However, in these problems, it is not enough to consider just the value of a minimum but also its flatness due to the fact that flat minima often represent a high generalization ability. In this paper, we introduce a collaborative neurodynamic optimization (CNO) method with a newly designed fitness function penalized by gradient norm (CNOGNP), which efficiently finds flat minima and thus enhances the generalization performance. Theoretical analysis proves that our CNOGNP algorithm is able to effectively find a flat minimum. Furthermore, simulations demonstrate that the proposed algorithm efficiently finds flat minima. Additionally, through classification experiments on Wine, Breast Cancer, Adult, CIFAR, and ImageNet datasets, we validate that the proposed CNOGNP improves the generalization compared with other algorithms.

## Overview of the Method

Our proposed method, CNOGNP, enhances the generalization of deep learning models by explicitly searching for **flat minima** in the loss landscape. It integrates a gradient norm penalty into the fitness function of the Collaborative Neurodynamic Optimization (CNO) framework.

The core idea is to modify the fitness function `φ(z)` to not only minimize the loss `f(z)` but also its sharpness, which is approximated by the squared L2 norm of the gradient `||∇f(z)||₂²`:

`φ(z) = f(z) + λ ||∇f(z)||₂²`

where `λ` is a hyperparameter balancing the two objectives. This approach leverages the powerful local and global search capabilities of CNO to efficiently converge to flatter regions, which are known to generalize better to unseen data.



Our proposed CNOGNP consistently outperforms baseline and state-of-the-art methods across various datasets and model architectures, demonstrating its superior generalization ability.

**Test Accuracies on CIFAR-10 and CIFAR-100 (Mean ± Std. Dev.):**

| Model     | Dataset  | Pretrain | SGD           | SAM           | CNO           | **CNOGNP (ours)**      |
| :-------- | :------- | :------- | :------------ | :------------ | :------------ | :--------------------- |
| ResNet20  | CIFAR10  | 92.61    | 92.68±0.08    | 92.61±0.03    | 92.68±0.03    | **92.71±0.07**         |
| ResNet32  | CIFAR10  | 93.52    | 93.52±0.08    | 93.53±0.11    | 93.56±0.01    | **93.57±0.09**         |
| ViT-T     | CIFAR100 | 53.34    | 53.82±0.04    | 53.79±0.16    | 53.81±0.10    | **53.88±0.02**         |

For more detailed results and analysis, please refer to our paper.
