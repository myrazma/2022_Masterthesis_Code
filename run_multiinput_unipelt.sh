pelt_method="unipelt_apl"
# UniPELT APL
if [ $pelt_method == "unipelt_apl" ]; then
    echo "Using Unipelt APL (adapter, prefix-tuning, lora; exclude: BitFit)"
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_unified_apl_bert
    add_enc_prefix=True
    train_adapter=True
    add_lora=True
    tune_bias=False
fi

# call the python file with stated parameters
python run_emp.py \
    --data_dir data/ \
    --output_dir output/unipelt_output  \
    --overwrite_output_dir \
    --model_name_or_path bert-base-uncased \
    --do_predict False \
    --do_eval True \
    --do_train True \
    --num_train_epochs 15 \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --early_stopping_patience 5 \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy no \
    --wandb_entity ${wandb_entity} \
    --use_tensorboard False\
    --tensorboard_output_dir ${tensorboard_output_dir} \
    --add_enc_prefix ${add_enc_prefix} \
    --train_adapter ${train_adapter} \
    --add_lora ${add_lora} \
    --tune_bias ${tune_bias} \
    --learning_rate ${learning_rate} \