rootdir=output_no_trainer_iterative/output_iterative_train_stateful_mnli
ckpt_path=output_no_trainer_iterative/output_iterative_train_stateful_mnli/roberta-base_lr_5e-4_div_2_sd_1111_epoch_6_warmup_prop_0_start_with_pos_embed_ln_clf 
ckpt_path=output_full_mnli
ckpt_path=${rootdir}/roberta-base_lr_2e-5_div_2_sd_1111_epoch_3_warmup_prop_0_start_with_pos_embed_ln_clf_0decay
rm ${ckpt_path}/norm_dict_all.log
for path in ${ckpt_path}/*
#for path in ${ckpt_path}/*roberta*
do
    if [[ -d $path ]]; then
        #echo "now evaluating $(basename -- $path)" >> ${ckpt_path}/norm_dict_all.log
        echo "now evaluating $(basename -- $path)" >> ${path}/norm_dict_all.log
        #python read_ckpt_norm.py --ckpt_path ${path} >> ${ckpt_path}/norm_dict_all.log
        python read_ckpt_norm.py --ckpt_path ${path} >> ${path}/norm_dict_all.log
    else
        echo "${path} is a file, skip...."
    fi
done
