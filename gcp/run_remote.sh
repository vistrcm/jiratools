#!/usr/bin/env bash
set -x
set -v
set -e

JOB_NAME=jiratrain_lin_simple_02
REGION=us-central1

BUCKET_PATH=$BUCKET_NAME/ml

#gsutil cp -r $(pwd)/../dump/train.csv gs://$BUCKET_PATH/data/
#gsutil cp -r $(pwd)/../dump/eval.csv gs://$BUCKET_PATH/data/

TRAIN_DATA=gs://$BUCKET_PATH/data/train.csv
EVAL_DATA=gs://$BUCKET_PATH/data/eval.csv


OUTPUT_PATH=gs://$BUCKET_PATH/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.8 \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 10000 \
    --eval-steps 100 \
    --verbosity DEBUG