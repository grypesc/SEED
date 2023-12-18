#!/bin/bash

if [ "$1" != "" ]; then
    echo "Running on gpu: $1"
else
    echo "No gpu has been assigned."
fi

SEED=$2

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"


# for SEED in 1 2 3
# do
for NUM_EXPERTS in 1 #3 5 11 1
do
PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --gpu $1 \
  --approach berg --gmms 1 --max-experts $NUM_EXPERTS --use-multivariate --ft-selection-strategy softmax  --nepochs 200 \
  --tau 3 --batch-size 256 --num-workers 8 --datasets imagenet_subset_kaggle --num-tasks 11 --nc-first-task 50 \
  --lr 0.05 --weight-decay 5e-4 --clipping 1 --alpha 0.95 --use-test-as-val --network resnet18 --extra-aug fetril \
  --momentum 0.9 --exp-name final_50+10x5_experts:${NUM_EXPERTS}_seed:$SEED --seed $SEED
done
# done
