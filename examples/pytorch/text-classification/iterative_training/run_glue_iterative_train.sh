
#where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.
#| Task  | Metric                       | Result      | Training time |
#|-------|------------------------------|-------------|---------------|
#| CoLA  | Matthew's corr               | 56.53       | 3:17          |
#| SST-2 | Accuracy                     | 92.32       | 26:06         |
#| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          |
#| STS-B | Person/Spearman corr.        | 88.64/88.48 | 2:13          |
#| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       |
#| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       |
#| QNLI  | Accuracy                     | 90.66       | 40:57         |
#| RTE   | Accuracy                     | 65.70       | 57            |
#| WNLI  | Accuracy                     | 56.34       | 24            |
#TASK_NAME=rte
#model="bert-base-cased"
source activate ../env

REPO_HOME=../
CACHE_DIR=$REPO_HOME/cache
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_DATASETS_CACHE=${CACHE_DIR}
export HF_METRICS_CACHE=${CACHE_DIR}
export TRANSFORMERS_OFFLINE=1
export TASK_NAME=$1
model="bert-large-uncased"
#model="t5-large"
times_lr_decay=2
lr=5e-4
#lr=2e-5
noised_alpha=0
#TASK_NAME="hans"
train_config="train"
validation_config="validation"
num_train_epochs=3
num_warmup_steps_prop=0.01
#TASK_NAME="anli"
#train_config="train_r1"
#validation_config="dev_r1"
#TASK_NAME="sst2"
#num_train_epochs=6
# export HF_HOME="./cache"


#for TASK_NAME in "wnli" "cola" "rte" "qnli" "qqp"
#for TASK_NAME in "sst2" "mnli" "qqp"
#for TASK_NAME in "qqp" "qnli"
#for TASK_NAME in "mnli" "qqp" "qnli"
#for TASK_NAME in "qqp" "mnli"
#for TASK_NAME in "qqp"
#for TASK_NAME in "sst2" "mnli"
#for num_warmup_steps_prop in 0
#for TASK_NAME in "mnli"
#for lr in 5e-4 2e-4 1e-4
for seed in 0 4 16 64 256
#for seed in 1024 2021 4096 6666
#for TASK_NAME in "qqp"
do
    #python run_glue_no_trainer_no_accelerator.py \
      #--model_name_or_path ${model} \
      #--task_name $TASK_NAME \
      #--max_length 128 \
      #--per_device_train_batch_size 32 \
      #--learning_rate 2e-5 \
      #--output_dir ./output_no_accelerator/output_large_bsz_long_clf_$TASK_NAME/${model} \
      #--num_train_epochs 3 \
    python run_glue_no_trainer_iterative_train.py \
      --model_name_or_path ${model} \
      --task_name $TASK_NAME \
      --restart_per_iteration \
      --max_length 128 \
      --per_device_train_batch_size 16 \
      --learning_rate ${lr} \
      --times_lr_decay ${times_lr_decay} \
      --weight_decay 0.01 \
      --seed ${seed} \
      --noised_alpha ${noised_alpha} \
      --num_warmup_steps_prop ${num_warmup_steps_prop} \
      --num_train_epochs ${num_train_epochs} \
      --train_config ${train_config} \
      --validation_config ${validation_config} \
      --output_dir ./output_no_trainer_iterative_restart/output_iterative_train_stateful_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay}_sd_${seed}_epoch_${num_train_epochs}_warmup_prop_${num_warmup_steps_prop}
      #--output_dir ./output_no_trainer_iterative/output_iterative_train_stateful_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay}_sd_${seed}_epoch_${num_train_epochs}_warmup_prop_${num_warmup_steps_prop}_noised_alpha_${noised_alpha}_start_with_pos_embed_ln_clf
      #--output_dir ./output_no_trainer_iterative/output_iterative_train_stateful_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay}_sd_${seed}_epoch_${num_train_epochs}_warmup_prop_${num_warmup_steps_prop}_noised_alpha_${noised_alpha}_start_with_pos_embed_type_ln_clf
      #--output_dir ./output_no_trainer_iterative/debug_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay} \
      #--output_dir ./output_no_trainer_iterative/output_optim_no_maintain_scheduler_reinit_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay} \
    #python run_glue_no_trainer_iterative_train.py \
      #--model_name_or_path ${model} \
      #--task_name $TASK_NAME \
      #--max_length 128 \
      #--per_device_train_batch_size 32 \
      #--learning_rate 5e-3 \
      #--output_dir ./output_no_trainer_iterative/output_lglr_scheduler_reinit_div2_$TASK_NAME/${model} \
      #--num_train_epochs 5
      #--num_train_epochs 96 \
      #--per_device_train_batch_size 32 #\
      #--gradient_accumulation_steps 4
    #
      #--output_dir ./output_clf_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 16
#
    #python run_glue_no_trainer2.py \
      #--model_name_or_path ${model} \
      #--task_name $TASK_NAME \
      #--max_length 128 \
      #--per_device_train_batch_size 128 \
      #--learning_rate 5e-5 \
      #--num_train_epochs 96 \
      #--output_dir ./output_large_bsz_long_layernorm_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 4
    #
      #--output_dir ./output_layernorm_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 16
#
    #python run_glue_no_trainer3.py \
      #--model_name_or_path ${model} \
      #--task_name $TASK_NAME \
      #--max_length 128 \
      #--per_device_train_batch_size 128 \
      #--learning_rate 5e-5 \
      #--num_train_epochs 96 \
      #--output_dir ./output_large_bsz_long_clf_ln_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 4
    #
      #--output_dir ./output_clf_ln_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 16
#
    #python run_glue_no_trainer4.py \
      #--model_name_or_path ${model} \
      #--task_name $TASK_NAME \
      #--max_length 128 \
      #--per_device_train_batch_size 128 \
      #--learning_rate 5e-5 \
      #--num_train_epochs 96 \
      #--output_dir ./output_large_bsz_long_clf_ln_pos_$TASK_NAME/${model} \
      #--gradient_accumulation_steps 4
done
      #--learning_rate 2e-5 \
      #--num_train_epochs 3 \
