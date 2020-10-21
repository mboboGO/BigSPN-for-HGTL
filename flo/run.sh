#!/bin/bash

nvidia-smi

cd ..

MODEL=dgtl
DATA=flo
BACKBONE=resnet101
SAVE_PATH=/output/

mkdir -p ${SAVE_PATH}

python main_dgtl.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 250 --lr 0.001 --epochs 90 --is_fix --pretrained &> ${SAVE_PATH}/fix.txt
python main_dgtl.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 24 --lr 0.0001 --epochs 90 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.txt

