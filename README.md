# 2022_Masterthesis_Code
The Code for my master thesis.


# models
The models can be found in the directory model/.

The model_utils class has model shared methods and class like the RegressionHead and training / evaluation methods.

# run

## Docker

Build the Dockerfile and run in bash.
```
docker build -t <docker-name> .
```

To run in bash and use gpus:

```
docker run --gpus all --rm -it -v "$PWD":/mzmarsly <docker-name> bash
```

## Parameter setting
While calling the model in bash, you can enter specific hyperparameters of the model. It should have the form of f.e. "--epochs 10", where the parameter name has "--" as a prefix. Possible parameters with example settings are:

| setting         | 	example command	| explanation   |
|---------------|-----------|---------------------------|
| epochs          | --epochs 10      |  Set the epoch number for training                   |
| learning_rate          | --learning_rate 2e-5   |  Set the learning rate for training                  |
|   empathy_type     |    --distress    |  Set either to --distress or --empathy. Default --empathy                  |
|  seed  |   --seed 42     |     Set the seed                 | 
| batch_size   | --batch_size 32      |         Set the batch size for training            | 
|  bert_type  |  --bert_type roberta-base     |   Set the bert model (load pretrained)                  | 
|  train_only_bias  |  --train_only_bias mlp (or --train_only_bias)     |  Train only the bias parameters of the model and freeze all other parameters. Will only be used in baseline_BERT. Options: "mlp": trainonly the bias in the mlp layers; "all": train all bias terms in the model, this is the default if the arg gets no input (see example)                | 
| adapter_type   |  --adapter_type pfeiffer     |   The adapter type to use. Will onyl be used in adapter_ownhead_BERT. Hand a string of one of the predefined adapter possibilities.                  | 
| model_name   | --model_name model1      |  The model name will be used as suffix for storing. Default: Timestamp with date and time of computer.                | 
| save_settings   | --save_settings      |  Either True or False, Default: False. If True, the settings of the model will be saved as a json.                   | 
| early_stopping   | --early_stopping      |   Either True or False, Default: False. If True, the model will use early stopping and save the model with the best correlation on de set.                  | 
| weight_decay   |  --weight_decay 0.1     |  Set the weigth decay of AdamW optimizer.                   | 
| save_model   |  --save_model     |   Either True or False, Default: False. If True, the model will be saved.                  | 
|  scheduler  |  --scheduler     |  Either True or False, Default: False. If True, the scheduler will be used.                   | 
|  activation  |  --activation tanh     |   Options: tanh, relu. Default: relu. Sets the activation function in the model                  | 
|  dropout  |   --dropout 0.2    |   Sets the dropout layers in the model. Default: 0.2.                  | 
| kfold  | --kfold 10  | Use kfold cross validation if kfold higher than 0. Default: 0, no cross validation used and use regular training.  |
