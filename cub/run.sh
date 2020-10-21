#!/bin/bash

nvidia-smi

cd ..

MODEL=dgtl
DATA=cub
BACKBONE=resnet101
SAVE_PATH=/output/

mkdir -p ${SAVE_PATH}

python main_dgtl.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 128 --lr 0.001 --epochs 90 --is_fix --pretrained &> ${SAVE_PATH}/fix.txt
python main_dgtl.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --lr 0.0001 --epochs 90 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.txt

