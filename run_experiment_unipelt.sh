# runs the model for empathy and distress
use_tensorboard=False
wandb_entity="masterthesis-zmarsly"
wandb_project="Results"

# -------- UniPELT setup --------
# UniPELT Setup: APL
pelt_method="full"
# if you wish to only run one task here, remove the other one fomr the list
task_names=( distress empathy )
# if you want to overwrite the learning rate in ANY case, do:
#overwrite_learning_rate=1e-4  

store_run=False
do_predict=True
train_ff_layers=False

#
for task_name in "${task_names[@]}"
do
    # The settings for the different methods
    # UniPELT
    if [ $pelt_method == "unipelt_aplb" ]; then
        echo "Using Unipelt (Prefix, adapter, lora, bitfit)"
        learning_rate=5e-4
        model_name=unipelt_aplb
        add_enc_prefix=True
        train_adapter=True
        add_lora=True
        tune_bias=True
    fi
    
    if [ $pelt_method == "unipelt_apl" ]; then
        echo "Using Unipelt APL (adapter, prefix-tuning, lora; exclude: BitFit)"
        learning_rate=1e-4
        model_name=unipelt_apl
        add_enc_prefix=True
        train_adapter=True
        add_lora=True
        tune_bias=False
    fi

    # UniPELT AP
    if [ $pelt_method == "unipelt_ap" ]; then
        echo "Using Unipelt APL (adapter, prefix-tuning; exclude: LoRA, BitFit)"
        learning_rate=5e-4
        model_name=unipelt_ap
        add_enc_prefix=True
        train_adapter=True
        add_lora=False
        tune_bias=False
    fi       
    
    # UniPELT AL
    if [ $pelt_method == "unipelt_al" ]; then
        echo "Using Unipelt APL (adapter, LoRA; exclude: prefix-tuning, BitFit)"
        learning_rate=5e-4
        model_name=unipelt_al
        add_enc_prefix=False
        train_adapter=True
        add_lora=True
        tune_bias=False
    fi

    # LoRA
    if [ $pelt_method == "lora" ]; then
        echo "Using LoRA"
        learning_rate=5e-4
        model_name=lora
        add_enc_prefix=False
        train_adapter=False
        add_lora=True
        tune_bias=False
    fi  

    # Adapters
    if [ $pelt_method == "adapter" ]; then
        echo "Using adapter"
        learning_rate=1e-4
        model_name=adapter
        add_enc_prefix=False
        train_adapter=True
        add_lora=False
        tune_bias=False
    fi

    # Full fine tuning
    if [ $pelt_method == "full" ]; then
        echo "Using Full fine tuning"
        learning_rate=2e-5
        model_name=full_fine_tuning
        add_enc_prefix=False
        train_adapter=False
        add_lora=False
        tune_bias=False
    fi

    # prefix Tuning
    if [ $pelt_method == "prefix" ]; then
        echo "Using Prefix-tuning"
        learning_rate=2e-4
        model_name=prefix
        add_enc_prefix=True
        train_adapter=False
        add_lora=False
        tune_bias=False
    fi

    # BitFit
    if [ $pelt_method == "bitfit" ]; then
        echo "Using BitFit"
        learning_rate=1e-3
        model_name=bitfit
        add_enc_prefix=False
        train_adapter=False
        add_lora=False
        tune_bias=True
    fi

    # Full fine tuning
    if [ $pelt_method == "feedforward" ]; then
        echo "Using Feed forward fine tuning"
        learning_rate=1e-4
        model_name=feedforward
        add_enc_prefix=False
        train_adapter=False
        add_lora=False
        tune_bias=False
        train_ff_layers=True # Only fine tuning of Forwardlayers
    fi

    # overwrite learning rate if desired and specified in the beginning
    # if overwrite learning rate is not Null
    if [ ! -z "$overwrite_learning_rate" ]; then
        learning_rate=$overwrite_learning_rate
        model_name="${model_name}_${learning_rate}"
    fi

    # -------- Multiinput setup --------
    # PCA setup
    dim=3
    data_lim=1000
    use_freq_dist=True
    freq_thresh=0.000005
    vocab_type=mm
    vocab_size=10
    use_question_template=False

    # Multiinput model setup
    use_pca_features=False
    use_lexical_features=False
    use_mort_features=False
    use_mort_article_features=False
    # None means using all
    # for distress: 04 (using pc 1 and 5)
    # for empathy: 24 (using pc 3 and 5)
    #mort_princ_comp=None
    if [ $task_name == "distress" ];
    then  # distress
        mort_princ_comp='04'
    else  # empathy
        mort_princ_comp='24'
    fi

    # -------- Additional Adapters input --------
    trained_adapter_dir="data/trained_adapters"
    # Stacking adapter (emotion most likely)
    stacking_adapter="bert-base-uncased-pf-emotion" # "AdapterHub/bert-base-uncased-pf-emotion"
    use_stacking_adapter=False
    train_all_gates_adapters=False

    # Multi task adapter
    # Add the adapter of the other task to the model
    use_sidetask_adapter=False

    # Sequential tansfer learning adapter
    pre_trained_sequential_transfer_adapter=None # "bert-base-uncased-pf-emotion"


    # -------- Rename based on variables --------


    if [ $use_pca_features == True ]; then
        model_name="${model_name}_pca${dim}"
    fi
    if [ $use_lexical_features == True ]; then
        model_name="${model_name}_lexical"
    fi
    if [ $use_mort_features == True ]; then
        model_name="${model_name}_MoRT-ess"
        if [ $mort_princ_comp != None ]; then
            model_name="${model_name}${mort_princ_comp}"
        fi
    fi
    if [ $use_mort_article_features == True ]; then
        model_name="${model_name}_MoRT-art"
        if [ $mort_princ_comp != None ]; then
            model_name="${model_name}${mort_princ_comp}"
        fi
    fi
    if [ $use_stacking_adapter == True ]; then
        model_name="${model_name}_stack"
    fi
    if [ $use_sidetask_adapter == True ]; then
        model_name="${model_name}_multitask"
    fi
    if [ $pre_trained_sequential_transfer_adapter != None ]; then
        model_name="${model_name}_sequential_tuning"
    fi

    tensorboard_output_dir="runs/${model_name}"
    model_name="${model_name}/${task_name}"
    output_dir="output/${model_name}"

    # for testing. if not delete:
    # max_train_samples
    # max_val_samples
    # set num_train_epochs to 15 again
    
    
    

    # call the python file with stated parameters
    python model/unipelt_model.py \
        --task_name ${task_name} \
        --data_dir data/ \
        --output_dir ${output_dir}  \
        --overwrite_output_dir \
        --model_name_or_path bert-base-uncased \
        --do_predict ${do_predict} \
        --do_eval True \
        --do_train True \
        --num_train_epochs 3 \
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
        --use_mort_features ${use_mort_features} \
        --use_mort_article_features ${use_mort_article_features} \
        --mort_princ_comp ${mort_princ_comp} \
        --dim ${dim} \
        --data_lim ${data_lim} \
        --use_freq_dist ${use_freq_dist} \
        --freq_thresh ${freq_thresh} \
        --vocab_type ${vocab_type} \
        --vocab_size ${vocab_size} \
        --use_question_template ${use_question_template}  \
        --stacking_adapter ${stacking_adapter} \
        --use_stacking_adapter ${use_stacking_adapter} \
        --train_all_gates_adapters ${train_all_gates_adapters} \
        --use_sidetask_adapter ${use_sidetask_adapter} \
        --pre_trained_sequential_transfer_adapter ${pre_trained_sequential_transfer_adapter} \
        --train_ff_layers ${train_ff_layers}
done