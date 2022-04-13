#! /bin/bash

source activate ./env

REPO_HOME=/home1/xuezhema/projects/max/transformers/examples/pytorch/text-classification
CACHE_DIR=$REPO_HOME/cache
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_DATASETS_CACHE=${CACHE_DIR}
export HF_METRICS_CACHE=${CACHE_DIR}
export TRANSFORMERS_OFFLINE=1
export TASK_NAME=$2
#clip_value="0.1"
#model="bert-base-cased"
model=$1

grad_clip_data_save_period=20
seeds=(1 2 3 5 7 11 13 17 19 23 29 31 37 41 42 43 47 53 59 61 67 71 73 79 83 89 97 101 997 1021)

for clip_value in 0.01 0.05 0.1 0.5 1 5 1e5; do
for run in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
  seed=${seeds[${run}]}
  python run_glue.py \
    --model_name_or_path ${model} \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --seed ${seed} \
    --num_train_epochs 3 \
    --cache_dir ${CACHE_DIR} \
    --use_clip_trainer True \
    --use_group_grad_norm_clip True
    --max_clip_value ${clip_value} \
    --grad_clip_data_save_period ${grad_clip_data_save_period} \
    --output_dir ./output/pre_correction_${model}_${TASK_NAME}_group_clip_by_norm_${clip_value}/seed${seeds[${run}]}
#    --output_dir ./output/pre_correction_${TASK_NAME}_clip_value_${clip_value}_period_${grad_clip_data_save_period}/seed${seed}
    #--output_dir ./output/save_${TASK_NAME}
#    --use_grad_value_clip True \
wait
done
done
echo "Press 'q' to exit"
count=0
while : ; do
read -n 1 k <&1
if [[ $k = q ]] ; then
printf "\nQuitting from the program\n"
break
else
((count=$count+1))
printf "\nIterate for $count times\n"
echo "Press 'q' to exit"
fi
done
