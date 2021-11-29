export TASK_NAME=rte
clip_value=0.25

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --cache_dir ./cache \
  --use_clip_trainer True \
  --max_clip_value ${clip_value} \
  --output_dir ./output/save_${TASK_NAME}_clip_value_${clip_value}
  #--output_dir ./output/save_${TASK_NAME}
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