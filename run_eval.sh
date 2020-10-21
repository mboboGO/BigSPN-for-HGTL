#!/bin/bash

nvidia-smi

MODEL=dgtl
DATA=cub
BACKBONE=resnet101
SAVE_PATH=/output/


python eval.py -a ${MODEL} -d ${DATA} --backbone ${BACKBONE} -b 128 --lr 0.0001 --epochs 90 --resume dgtl_32.0331.model #&> #${SAVE_PATH}/ft.txt

