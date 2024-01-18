# Divide and not forget: Ensemble of selectively trained experts in Continual Learning: ICLR2024 (Main track)

![image](inference.jpg?raw=true "inference")

This repository contains code for the SEED paper published at the main track of ICLR2024. It is based on FACIL (https://github.com/mmasana/FACIL) benchmark.
To reproduce results run one of provided scripts. 


Run SEED on CIFAR100 10 task, 10 classes each:
```bash
python cifar10x10.sh
```

Run SEED on CIFAR100 20 task, 5 classes each:
```bash
python cifar20x5.sh
```

Run SEED on CIFAR100 50 task, 2 classes each:
```bash
python cifar50x2.sh
```

If you would like to cooperate on improving the method, please contact me via LinkedIn or Facebook, I have several ideas.