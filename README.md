# Divide and not forget: Ensemble of selectively trained experts in Continual Learning: ICLR2024 (Main track)

https://arxiv.org/abs/2401.10191  
https://openreview.net/forum?id=sSyytcewxe  

![image](inference.jpg?raw=true "inference")

This repository contains code for the SEED paper published at the main track of ICLR2024. It is based on FACIL (https://github.com/mmasana/FACIL) benchmark.
To reproduce results run one of provided scripts. 

Setup environment according to readme of FACIL.

Run SEED on CIFAR100 10 tasks, 10 classes each:
```bash
bash cifar10x10.sh
```

Run SEED on CIFAR100 20 tasks, 5 classes each:
```bash
bash cifar20x5.sh
```

Run SEED on CIFAR100 50 tasks, 2 classes each:
```bash
bash cifar50x2.sh
```
To lower the number of parameters as in Tab.5 use ```--network resnet 20 --shared 2```. You can also add parameter pruning as in DER.

To reproduce results for ImageNet Subset download ImageNet subset from https://www.kaggle.com/datasets/arjunashok33/imagenet-subset-for-inc-learn and put it in ```../data``` directory.
```bash
bash imagenet.sh
```

To reproduce results for DomainNet download cleaned version from http://ai.bu.edu/M3SDA/ and put it in ```../data``` directory (unzip it).
Run SEED on DomainNet 12 tasks of different domains, 25 classes each:
```bash
bash domainnet12x25.sh
```

If you would like to cooperate on improving the method, please contact me via LinkedIn or Facebook, I have several ideas.

If you find this work useful, please consider citing it:

```
@article{rypesc2024divide,
  title={Divide and not forget: Ensemble of selectively trained experts in Continual Learning},
  author={Rype{\'s}{\'c}, Grzegorz and Cygert, Sebastian and Khan, Valeriya and Trzci{\'n}ski, Tomasz and Zieli{\'n}ski, Bartosz and Twardowski, Bart{\l}omiej},
  journal={arXiv preprint arXiv:2401.10191},
  year={2024}
   ```
