# 2022_Masterthesis_Code
The code for my master's thesis.

# Models

Overview

TODO: Update the available models
The models can be found in the directory [model](#model/).

The script [baseline_BERT.py](model/baseline_BERT.py) is implementing a baseline RoBERTa model with the possibility of BitFit training (by setting --train_only_bias in cmd parameters]. BitFit (BIas-Term FIne-Tuning: https://arxiv.org/abs/2106.10199) is a version of parameter-efficient training where only the bias in a pre-trained transformer model are trained. There is even the option to only train the bias parameters in the mlp of the transformers (no attention layers).

The script [adapter_ownhead_BERT.py](model/adapter_ownhead_BERT.py) is implementing a RoBERTa model with different parameter-efficient training methods like adapters and prefix tuning using the adapter hub.

The script model_utils.py has model shared methods and class like the RegressionHead and training / evaluation methods.

## The task: Empathy and distress prediction
The task is a supervised regression task to predict empathy and distress ratings from texts (Buechel et al, 2018). 

Particpants were asked to read 5 news articles, after each article they reported their empathy and distress level on a 14 item questionnaire and wrote a reaction-essay. This essay is the base / input for our NLP model. The label / target are the average questionnaire. For an example and more information about the data please look at the [data README.md](data/buechel_empathy/README.md).


# Running the Project

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

# Running the UniPELT and PELT methods
This code is using the submodule for the slightly modified UniPELT implementation, orignially implemented by Mao et. al., 2022. To use the [unipelt_model.py](model/unipelt_model.py) model, make sure that the submodule is at the newest commit, by [updating the submodule](#Submodules).

The gates can be returned during prediciton by setting the 'return_gates' variable to True. 
```
output, gates = trainer.predict(test_dataset=eval_dataset, return_gates=True)
```

To run the mode, use [run_experiment_unipelt.sh](run_experiment_unipelt.sh). The parameters for each method can be set here. We support the following settings and methods

The model can be changed in the bash script to run different PELT methods, adapter compositions using the different setups using the pelt_method variable.

## PELT and UniPELT
With *pelt_method*, you can set the tuning method using the following names:
| PELT method         | 	name	      | default learning rate |
|---------------  |----------------------  |----------------------  |
| Full fine-tuning  |  full | 2e-5 |
| Tuning feed-forward | feedforward | 1e-4 |
| BitFit  |  bitfit | 1e-3 |
| Bottleneck Adapter  |  adaper | 1e-4 |
| Prefix-tuning  |  prefix | 2e-4 |
| LoRA  |  lora | 5e-4 |
| UniPELT APL  |  unipelt_aplb | 5e-4 |
| UniPELT APLB  |  unipelt_apl | 1e-4 |
| UniPELT AL | unipelt_al | 5e-4 |
| UniPELT AP | unipelt_ap | 5e-4 |

## The Adapter Composition
The adapter composition has the following settings:


    trained_adapter_dir="data/trained_adapters"
    # Stacking adapter (emotion most likely)
    stacking_adapter="bert-base-uncased-pf-emotion" # "AdapterHub/bert-base-uncased-pf-emotion"
    use_stacking_adapter=False
    train_all_gates_adapters=False

    # Multi task adapter
    # Add the adapter of the other task to the model
    use_sidetask_adapter=False

    # Sequential tansfer learning adapter
    pre_trained_sequential_transfer_adapter=None

| variable        | 	example input	      | explanation |
|---------------  |----------------------  |----------------------  |
| trained_adapter_dir  | "data/trained_adapters"  | The directory with the adpater (already downloaded) |
| stacking_adapter  | "bert-base-uncased-pf-emotion"  | The adapter that should be used for stacking (e.g. Adapter Stack EMO) |
| use_stacking_adapter  | True  | Set to True to use the stack setup with the Adpater specified in *stacking_adapter* |
| train_all_gates_adapters  | True  | If True, all gates will be trained in the Stacking setup. (Set True for the experiment)|
| pre_trained_sequential_transfer_adapter  | "bert-base-uncased-pf-emotion"  | Use this adapter for sequential tuning of the adapter (Adapter Seq EMO). If you do not want to use sequential tuning, use set it to None. |

## Multi-iput
The different features can be concatenated to the classification head of the model by setting the following variables:
| variable        | 	example input	      | explanation |
|---------------  |----------------------  |----------------------  |
| use_pca_features  |  True | If set to True, the PCA features will be used (ED/DD). |
| use_lexical_features  |  True | If set to True, the lexical features will be used. |
| use_mort_features  |  True | If set to True, the MD of the essay will be used. |
| use_mort_features  |  True | If set to True, the MD of the article will be used. |


# Running the Empathy (ED) and Distress Direction (DD)

The fittet PCA (sklearn) for the both directions (pca_ED.p and pca_DD.p) can be found in [EmpDim/pca_projections/](EmpDim/pca_projections/)

To use the PCA, the input has to be transformed using Sentence-BERT (Reimers & Gurevych, 2019) with 'bert-large-nli-mean-tokens' as transformer model.

Alternatively, the analysis can be run with the script *run_empdim.sh*. This will run the code in [pca.py](EmpDim/pca.py) You can hereby change the following settings: 


| setting         | 	example command	        | explanation   |
|---------------  |----------------------     |---------------------------|
| dim          | --dim 3               |  The n components of the PCA                   |
| use_fdist  | --use_fdist True     |  Whether removing words based on the frequency distribution should be activated or not                  |
| freq_thresh    | --freq_thresh 0.000005                | The threshold for the frequency distribution. |
| vocab_type | --vocab_type mm | The vocabulary type, can be 'mm' (min and max scores), 'mmn' (min, max neutral scores), 'range' (word from the whole range of scores) |
| vocab_size | --vocab_size 10 | The size of the vocabulary per setting, e.g., for mm choose 10 min and 10 max. For range, this number is accordingly for each bin (0.1). |
| use_question_template | --use_question_template True | Whether to use the template or not |
| task_name | --task_name empathy | The prediction task, e.g., empathy or distress|


# Running Moral Direction
Since the moral direction has other dependencies, we need to use another Docker image:

Build the Dockerfile
```
docker build -t <docker-name> -f DockerfileMoRT .
```
The pca for the moral direction is in *data/MoRT_projection*.


# Generate output format for CodaLab
Using the script *output_formatter.py* in utils, you can input a model name. 
```
python3 output_formatter.py adapter
```
The model has to be in the output folder and already generated some results (as seen in the output foler).
The form should be as follows: 
- output/model_name/empathy/
- output/model_name/distress/
and containing the file *test_results_distress.txt* (or empathy).
This file is generated, setting the parameter *do_predict* to True.


# Bibliography
Mao, Y., Mathias, L., Hou, R., Almahairi, A., Ma, H., Han, J., Yih, S., & Khabsa, M. (2022). Unipelt: A unified framework for parameter-efficient language model tuning. Proceedings of the 60th Annual Meeting of the Association for Computational Lin- guistics (Volume 1: Long Papers), 6253–6264. https://aclanthology.org/2022.acl- long.433/

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese bert-networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992. https://aclanthology.org/D19-1410/
