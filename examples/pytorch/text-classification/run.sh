export TASK_NAME="rte"
clip_value="99999"
#model="bert-base-cased"
model="roberta-base"

grad_clip_data_save_period=20
#for seed in 1 2 3 4 5
for seed in 2
do
  python run_glue.py \
    --model_name_or_path ${model} \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --seed ${seed} \
    --num_train_epochs 3 \
    --cache_dir ./cache \
    --use_clip_trainer True \
    --use_group_grad_norm_clip True \
    --max_clip_value ${clip_value} \
    --grad_clip_data_save_period ${grad_clip_data_save_period} \
    --output_dir ./output/pre_correction_${model}_${TASK_NAME}_group_clip_by_norm_${clip_value}/seed${seed}
#    --output_dir ./output/pre_correction_${TASK_NAME}_clip_value_${clip_value}_period_${grad_clip_data_save_period}/seed${seed}
    #--output_dir ./output/save_${TASK_NAME}
#    --use_grad_value_clip True \
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