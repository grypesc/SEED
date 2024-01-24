# Divide and not forget: Ensemble of selectively trained experts in Continual Learning: ICLR2024 (Main track)

https://arxiv.org/abs/2401.10191  
https://openreview.net/forum?id=sSyytcewxe  

![image](inference.jpg?raw=true "inference")

This repository contains code for the SEED paper published at the main track of ICLR2024. It is based on FACIL (https://github.com/mmasana/FACIL) benchmark.
To reproduce results run one of provided scripts. 


Run SEED on CIFAR100 10 tasks, 10 classes each:
```bash
python cifar10x10.sh
```

Run SEED on CIFAR100 20 tasks, 5 classes each:
```bash
python cifar20x5.sh
```

Run SEED on CIFAR100 50 tasks, 2 classes each:
```bash
python cifar50x2.sh
```
To lower the number of parameters as in Tab.5 use ```--network resnet 20 --shared 2```. You can also add parameter pruning as in DER.

If you would like to cooperate on improving the method, please contact me via LinkedIn or Facebook, I have several ideas.
