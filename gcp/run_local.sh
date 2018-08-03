#!/usr/bin/env bash

TRAIN_DATA=$(pwd)/../dump/train.csv
EVAL_DATA=$(pwd)/../dump/eval.csv

MODEL_DIR=output

# rm -rf $MODEL_DIR/*

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG
