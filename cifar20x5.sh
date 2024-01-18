#!/bin/bash
for NUM_EXPERTS in 5
do
  python src/main_incremental.py --approach seed --gmms 1 --max-experts $NUM_EXPERTS --use-multivariate --nepochs 200 --tau 3 --batch-size 128 --num-workers 4 --datasets cifar100_icarl --num-tasks 20 --nc-first-task 5 --lr 0.05 --weight-decay 5e-4 --clipping 1 --alpha 0.99 --use-test-as-val --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name exp2 --seed 0
done

