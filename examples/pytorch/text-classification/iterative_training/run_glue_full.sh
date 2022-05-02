
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
num_train_epochs=3
#TASK_NAME="sst2"
TASK_NAME="mnli"
#TASK_NAME="hans"
train_config="train"
validation_config="validation"
#num_train_epochs=10
# anli only
#TASK_NAME="anli"
#train_config="train_r1"
#validation_config="dev_r1"

export HF_HOME="./cache"


#for TASK_NAME in "wnli" "cola" "rte" "qnli" "qqp"
#for TASK_NAME in "sst2" "mrpc" "stsb"
#for TASK_NAME in "qqp" "mnli"
#for seed in 0 4 16 64 256 1024 2021 4096 6666
for seed in 1111
#for TASK_NAME in "mrpc" "stsb"
do
    python run_glue_no_trainer_full.py \
      --model_name_or_path ${model} \
      --task_name $TASK_NAME \
      --max_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --seed ${seed} \
      --num_train_epochs ${num_train_epochs} \
      --output_dir ./output_full_$TASK_NAME/${model}_sd_${seed} \
      --train_config ${train_config} \
      --validation_config ${validation_config}
      #--output_dir ./output_full_$TASK_NAME/${model}_sd_${seed}
done
