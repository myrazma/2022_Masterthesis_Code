use_tensorboard=False
wandb_entity="masterthesis-zmarsly"

# UniPELT Setup: APL
pelt_method="unipelt_apl"
if [ $pelt_method == "unipelt_apl" ]; then
    echo "Using Unipelt APL (adapter, prefix-tuning, lora; exclude: BitFit)"
    learning_rate=5e-4
    tensorboard_output_dir=runs/multiinput_pelt_unified_apl_bert
    add_enc_prefix=True
    train_adapter=True
    add_lora=True
    tune_bias=False
fi


# PCA setup
task_name=distress
store_run=False
dim=1
data_lim=1000
use_freq_dist=True
freq_thresh=0.000005
vocab_type=mm
vocab_size=10

# Multiinput model setup
use_pca_features=True
if [ $use_pca_features == True ]; then
    tensorboard_output_dir=tensorboard_output_dir="${tensorboard_output_dir}_pca"
fi
use_lexical_features=True
if [ $use_lexical_features == True ]; then
    tensorboard_output_dir=tensorboard_output_dir="${tensorboard_output_dir}_lexical"
fi


# for testing. if not delete:
# max_train_samples
# max_val_samples
# set num_train_epochs to 15 again


# call the python file with stated parameters
python model/unipelt_model.py \
    --task_name ${task_name} \
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
    --use_tensorboard ${use_tensorboard}\
    --tensorboard_output_dir ${tensorboard_output_dir} \
    --add_enc_prefix ${add_enc_prefix} \
    --train_adapter ${train_adapter} \
    --add_lora ${add_lora} \
    --tune_bias ${tune_bias} \
    --learning_rate ${learning_rate} \
    --use_pca_features ${use_pca_features} \
    --use_lexical_features ${use_lexical_features} \
    --dim ${dim} \
    --data_lim ${data_lim} \
    --use_freq_dist ${use_freq_dist} \
    --freq_thresh ${freq_thresh} \
    --vocab_type ${vocab_type} \
    --vocab_size ${vocab_size} \