#!/bin/bash
python src/main_incremental.py --approach seed --gmms 1 --max-experts 5 --use-multivariate --nepochs 200 --tau 3 --batch-size 128 --num-workers 8 --datasets cifar100_icarl_224 --num-tasks 6 --nc-first-task 50 --lr 0.05 --weight-decay 1e-4 --clipping 1 --alpha 1 --ftepochs 0 --use-test-as-val --network resnet18 --extra-aug fetril --momentum 0.9 --exp-name table2 --seed 0


