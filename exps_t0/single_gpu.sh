#! /bin/bash
REPO_HOME=examples/pytorch/text-classification
source activate ${REPO_HOME}/env
# export TRANSFORMERS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
# export HF_DATASETS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
# export HF_METRICS_CACHE=/home/chuntinz/tir5/pretrain_models/huggingface
cache_dir=${REPO_HOME}/cache

# max cluster
CACHE_DIR=$REPO_HOME/cache
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_DATASETS_CACHE=${CACHE_DIR}
export HF_METRICS_CACHE=${CACHE_DIR}
export TRANSFORMERS_OFFLINE=1

DATE=`date +%Y%m%d`

dataset="anli"
subset="none"
bsz=100
testset_name="dev_r1"

dataset="super_glue"
subset="cb"
bsz=1
testset_name="validation"

test_mode="t0"
model="bigscience/T0_3B"
model="T0pp"

exp_name=${test_mode}.${dataset}.${testset_name}
SAVE=checkpoints/${dataset}/${DATE}/${exp_name}
rm -rf ${SAVE}; mkdir -p ${SAVE}

CUDA_VISIBLE_DEVICES=0 python -u examples/pytorch/t0-zero-shot/run_t0.py \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size 1  --per_gpu_eval_batch_size ${bsz} \
  --test_mode ${test_mode} --cache_dir ${CACHE_DIR} \
  --output_dir ${SAVE} --overwrite_output_dir --fp16 \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt
