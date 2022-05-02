
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
export HF_HOME="./cache"


#for TASK_NAME in "wnli" "cola" "rte" "qnli" "qqp"
#for TASK_NAME in "sst2" "mrpc" "stsb"
#for TASK_NAME in "qqp" "mnli"
#for TASK_NAME in "cola" "rte" "qnli" "mnli" 
for TASK_NAME in "mnli"
#for TASK_NAME in "sst2" "mrpc" "stsb" "qqp"
do
    python run_glue_no_trainer_layer.py \
      --model_name_or_path ${model} \
      --task_name $TASK_NAME \
      --max_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 5e-5 \
      --num_train_epochs 6 \
      --output_dir ./output_clf_layernorm_layer11_$TASK_NAME/${model}
done
