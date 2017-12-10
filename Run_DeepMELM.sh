#!/bin/bash
# Usage:
# 
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1 
DATASET=$2    #VOC2007
BASE_MODEL=$3 #VGG16
PROPOSALS=$4  #SSW
Rate1=$5
Rate2=$5
Recurrent_Pattern=$6 #--Anneal  --None
test_epoch_num=20
Postfix=$7


Folder="data/results/${DATASET}/${BASE_MODEL}/ARL-${Rate1}-${Rate2}-${Recurrent_Pattern}-${PROPOSALS}"
mkdir -p "${Folder}-${Postfix}/"

LOG="${Folder}-${Postfix}/log_train.txt"
echo Logging output to "$LOG"
th utils/train.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix |tee $LOG


LOG="${Folder}-${Postfix}/log_test.txt"
echo Logging output to "$LOG"
th utils/test.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix |tee $LOG



th utils/detection_mAP.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix "${Folder}-${Postfix}/scores_test_epoch${test_epoch_num}.h5" 1
th utils/detection_mAP.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix "${Folder}-${Postfix}/scores_test_epoch${test_epoch_num}.h5" 2
th utils/detection_mAP.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix "${Folder}-${Postfix}/scores_test_epoch${test_epoch_num}.h5" 3
th utils/detection_mAP.lua $GPU_ID $DATASET $BASE_MODEL $PROPOSALS $Rate1 $Rate2 $Recurrent_Pattern $test_epoch_num $Postfix "${Folder}-${Postfix}/scores_test_epoch${test_epoch_num}.h5" 2+3
