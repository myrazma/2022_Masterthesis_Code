# 2022_Masterthesis_Code
The code for my master's thesis.

# Models
TODO: Update the available models
The models can be found in the directory [model](model).

The script [baseline_BERT.py](model/baseline_BERT.py) is implementing a baseline RoBERTa model with the possibility of BitFit training (by setting --train_only_bias in cmd parameters]. BitFit (BIas-Term FIne-Tuning: https://arxiv.org/abs/2106.10199) is a version of parameter-efficient training where only the bias in a pre-trained transformer model are trained. There is even the option to only train the bias parameters in the mlp of the transformers (no attention layers).

The script [adapter_ownhead_BERT.py](model/adapter_ownhead_BERT.py) is implementing a RoBERTa model with different parameter-efficient training methods like adapters and prefix tuning using the adapter hub.

The script model_utils.py has model shared methods and class like the RegressionHead and training / evaluation methods.


# Running the Code

## Docker

Build the Dockerfile
```
docker build -t <docker-name> .
```

and run in bash (using gpus here, container will be removed after exit)

```
docker run --gpus all --rm -it -v "$PWD":/mzmarsly <docker-name> bash
```

Right now, you can run the code by calling one of the model scripts with desired parameters. A Docker container for running the code with any further adjustment will follow after the code experimentation phase of this part of the thesis. For example you can run **BitFit** with the following command

```
python model/baseline_BERT.py --train_only_bias --epochs 10 --learning_rate 1e-4
```

## Submodules
To run code from the submodules you need to init the submodules with 
```
git submodule init
```
and 
```
git submodule update
```

After initialization you can run
```
git submodule update --recursive --remote
```
for more updates of this submodule.

## Parameter setting

The model with the UniPELT and PELT methods can be called with the bash script 

| setting         | 	example command	        | explanation   |
|---------------  |----------------------     |---------------------------|
| epochs          | --epochs 10               |  Set the epoch number for training                  |
| learning_rate  | --learning_rate 2e-5      |  Set the learning rate for training                  |

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
    
    
    
| epochs          | --epochs 10               |  Set the epoch number for training                   |
 | learning_rate  | --learning_rate 2e-5      |  Set the learning rate for training                  |
| empathy_type    | --distress                |  Set either to --distress or --empathy. Default --empathy                  |
|  seed           | --seed 42                 |     Set the seed                 | 
| batch_size      | --batch_size 32           |         Set the batch size for training            | 
|  bert_type      | --bert_type roberta-base  |   Set the bert model (load pretrained)                  | 
| train_only_bias | --train_only_bias mlp (or --train_only_bias) |  Train only the bias parameters of the model and freeze all other parameters. Will only be used in baseline_BERT. Options: "mlp": trainonly the bias in the mlp layers; "all": train all bias terms in the model, this is the default if the arg gets no input (see example)               | 
| adapter_type    | --adapter_type pfeiffer   |   The adapter type to use. Will onyl be used in adapter_ownhead_BERT. Hand a string of one of the predefined adapter possibilities.             | 
| model_name      | --model_name model1       |  The model name will be used as suffix for storing. Default: Timestamp with date and time of computer.                | 
| save_settings   | --save_settings           |  Either True or False, Default: False. If True, the settings of the model will be saved as a json.                   | 
| early_stopping  | --early_stopping          |   Either True or False, Default: False. If True, the model will use early stopping and save the model with the best correlation on de set.      | 
| weight_decay    | --weight_decay 0.1        |  Set the weigth decay of AdamW optimizer.                   | 
| save_model      | --save_model              |   Either True or False, Default: False. If True, the model will be saved.                  | 
|  scheduler      | --scheduler               |  Either True or False, Default: False. If True, the scheduler will be used.                   | 
|  activation     | --activation tanh         |   Options: tanh, relu. Default: relu. Sets the activation function in the model                  | 
|  dropout        | --dropout 0.2             |   Sets the dropout layers in the model. Default: 0.2.                  | 
| kfold           | --kfold 10                | Use kfold cross validation if kfold higher than 0. Default: 0, no cross validation used and use regular training.  |


# Running the Empathy (ED) and Distress Direction (DD)

The fittet PCA for the both directions can be found in EmpDim/pca_models/

The input texts has to be transformed with Sentence-BERT (Reimers & Gurevych, 2019) with 'bert-large-nli-mean-tokens' as transformer model.

You can also run the analsis by running the script *run_empdim.sh* with the desired settings.

# Running Moral Direciotn
Since the moral direction has other dependencies, we need to use another Docker image:

Build the Dockerfile
```
docker build -t <docker-name> -f DockerfileMoRT .
```

# Generate output format for CodaLab
Using the script *output_formatter.py* in utils, you can input a model name. The model has to be in the output folder and already generated some results (as seen in the output foler).
The form should be as follows: 
- output/model_name/empathy/
- output/model_name/distress/
and containing the file *test_results_distress.txt* (or empathy).
This file is generated, setting the parameter *do_predict* to True.


## Run bash script
TODO

# Note
I'm still currently working on the code and trying new things, therefore cleaner, more structured code will follow for the scripts currently under construction :). Also I'm running the code on another device and commiting even small changes that might have a difference in model training.



# Bibliography
Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084. https://arxiv.org/pdf/1908. 10084.pdf
