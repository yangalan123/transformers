
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
model="roberta-base"
times_lr_decay=2
lr=5e-4
TASK_NAME="mnli"
noised_alpha=0
#TASK_NAME="hans"
train_config="train"
validation_config="validation"
num_train_epochs=3
#TASK_NAME="anli"
#train_config="train_r1"
#validation_config="dev_r1"
eval_log_name="eval_${TASK_NAME}.log"
EVAL_TASK_NAME="mnli"
#TASK_NAME="sst2"
seed=1111
num_train_epochs=1
export HF_HOME="./cache"


#for TASK_NAME in "wnli" "cola" "rte" "qnli" "qqp"
#for TASK_NAME in "sst2" "mnli" "qqp"
#for TASK_NAME in "qqp" "qnli"
#for TASK_NAME in "mnli" "qqp" "qnli"
#for TASK_NAME in "qqp" "mnli"
#for TASK_NAME in "qqp"
#for TASK_NAME in "sst2" "mnli"
for num_warmup_steps_prop in 0
#for TASK_NAME in "mnli"
#for lr in 5e-4 2e-4 1e-4
#for seed in 0 4 16 64 256 1024 2021 4096 6666
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
      --max_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate ${lr} \
      --compute_norm_per_layer \
      --times_lr_decay ${times_lr_decay} \
      --seed ${seed} \
      --no_training \
      --load_from_checkpoint \
      --noised_alpha ${noised_alpha} \
      --num_warmup_steps_prop ${num_warmup_steps_prop} \
      --num_train_epochs ${num_train_epochs} \
      --train_config ${train_config} \
      --validation_config ${validation_config} \
      --log_name ${eval_log_name} \
      --load_dir output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_2e-5_div_2_sd_1111_epoch_3_warmup_prop_0_start_with_pos_embed_ln_clf_0decay \
      --output_dir output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_2e-5_div_2_sd_1111_epoch_3_warmup_prop_0_start_with_pos_embed_ln_clf_0decay
      #--load_dir output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_2e-5_div_2_sd_1111_epoch_3_warmup_prop_0_start_with_pos_embed_ln_clf \
      #--output_dir output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_2e-5_div_2_sd_1111_epoch_3_warmup_prop_0_start_with_pos_embed_ln_clf 
      #--output_dir ./output_no_trainer_iterative/output_iterative_train_stateful_$TASK_NAME/${model}_lr_${lr}_div_${times_lr_decay}_sd_${seed}_epoch_${num_train_epochs}_warmup_prop_${num_warmup_steps_prop}_noised_alpha_${noised_alpha}_start_with_pos_embed_type_ln_clf 
      #--load_dir output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_5e-4_div_2_sd_1111_epoch_6_warmup_prop_0_start_with_embed_pos_type_ln_clf
done
      #--learning_rate 2e-5 \
      #--num_train_epochs 3 \
