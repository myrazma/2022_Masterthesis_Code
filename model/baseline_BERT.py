# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
#
# ------------------------------

# utils
from logging import root
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
# Transformers, torch and model utils
from transformers import BertModel, BertConfig, BertForSequenceClassification, AutoModel, RobertaModel
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from transformers import logging



# import own module
import model_utils

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils
import preprocessing

class BertRegressor(nn.Module):
    # source (changed some things): [2]  
    # https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    def __init__(self, bert_type="bert-base-uncased", train_only_bias=False, train_bias_mlp=False, activation_func='relu', dropout=0.5):
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1

        if bert_type == 'roberta-base':
            self.bert = RobertaModel.from_pretrained(bert_type)
        else:
            self.bert = BertModel.from_pretrained(bert_type)

        if train_only_bias == 'all' or train_only_bias == 'mlp':
            print(f'\n------------ Train only the bias: {train_only_bias} ------------\n')
            bias_filter = lambda x: 'bias' in x
            if train_only_bias == 'mlp':  # train only the mlp layer (excluding all biases in the attention layers)
                bias_filter = lambda x: 'bias' in x and not 'attention' in x

            names = [n for n, p in self.bert.named_parameters()]
            params = [param for param in self.bert.parameters()]
            for n, p in zip(names, params):
                if bias_filter(n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.regression_head = model_utils.RegressionHead(D_in=D_in, D_out=D_out, activation_func=activation_func, dropout=dropout)

        # get the size of the model parameters (head and bert separated)
        self.bert_parameter_count = model_utils.count_updated_parameters(self.bert.parameters())
        self.head_parameter_count = model_utils.count_updated_parameters(self.regression_head.parameters())

    def forward(self, input_ids, attention_masks):
        bert_outputs = self.bert(input_ids, attention_masks)
        outputs = self.regression_head(bert_outputs)
        return outputs


def pd_to_dataset(data_df):
    """Create hugginface dataset from pandas dataframe

   Args:
        data_df (pd.DataFrame): _description_

    Returns:
        Dataset: The huggingface dataset from datasets.Dataset
    """
    data_df = Dataset.from_pandas(data_df)
    return data_df


def pd_to_datasetdict(train, dev):
    """Create dataset dictionary with train and dev split from two pandas dataframes

    Args:
        data_df (pd.DataFrame): _description_
        data_dev (pd.DataFrame): _description_

    Returns:
        Datasetdict: The huggingface datasets dictionary from datasets.DatasetDict with train and dev
    """
    dataset_train = Dataset.from_pandas(train)
    dataset_dev = Dataset.from_pandas(dev)
    whole_dataset = DatasetDict({'train': dataset_train, 'dev': dataset_dev})
    return whole_dataset

def create_tensor_data(inputs, masks, labels):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    return dataset


def create_dataloaders(inputs, masks, labels, batch_size):
    # Source: [2]
    dataset = create_tensor_data(inputs, masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def run(settings, root_folder=""):

    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    #logging.set_verbosity_warning()
    #logging.set_verbosity_error()
    data_root_folder = root_folder + 'data/'
    # -------------------
    #     parameters
    # -------------------

    empathy_type = settings['empathy_type']
    bert_type = settings['bert-type']
    my_seed = settings['seed']
    batch_size = settings['batch_size']
    learning_rate = settings['learning_rate']
    epochs = settings['epochs']
    train_only_bias = settings['train_only_bias']
    weight_decay = settings['weight_decay']
    use_scheduler = settings['scheduler']
    use_early_stopping = settings['early_stopping']
    activation_func = settings['activation']
    dropout = settings['dropout']

    using_roberta = False
    if bert_type == 'roberta-base':
        using_roberta = True

    # -------------------
    #   load data
    # -------------------
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)
    
    # --- Create hugginface datasets ---
    # TODO: Use all data later on
    data_train = pd_to_dataset(data_train_pd[:10])
    data_dev = pd_to_dataset(data_dev_pd[:5])

    #  Create hugginface datasetsdict
    # data_train_dev = pd_to_datasetdict(data_train, data_dev)
        
    # -------------------
    #   preprocess data
    # -------------------

    # --- tokenize data ---
    if using_roberta:
        tokenizer = RobertaTokenizer.from_pretrained(bert_type)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type)

    data_train_encoded = data_train.map(lambda x: model_utils.tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: model_utils.tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)

    # --- shuffle data ---
    data_train_encoded_shuff = data_train_encoded.shuffle(seed=my_seed)
    data_dev_encoded_shuff = data_dev_encoded.shuffle(seed=my_seed)

    # get input_ids, attention_mask and labels
    # train
    input_ids_train = np.array(data_train_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_train_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_train = np.array(data_train_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_train = np.array(data_train_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    # dev
    input_ids_dev = np.array(data_dev_encoded_shuff["input_ids"]).astype(int)
    attention_mask_dev = np.array(data_dev_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_dev = np.array(data_dev_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_dev = np.array(data_dev_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)

    # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
    label_scaled_empathy_train = preprocessing.normalize_scores(label_empathy_train, (1,7))
    label_scaled_empathy_dev = preprocessing.normalize_scores(label_empathy_dev, (1,7))
    label_scaled_distress_train = preprocessing.normalize_scores(label_distress_train, (1,7))
    label_scaled_distress_dev = preprocessing.normalize_scores(label_distress_dev, (1,7))
    
    # --- create datasets ---
    # for empathy
    pytorch_dataset_emp_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_empathy_train)
    pytorch_dataset_emp_dev = create_tensor_data(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev)
    # for distress
    pytorch_dataset_dis_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_distress_train)
    pytorch_dataset_dis_dev = create_tensor_data(input_ids_dev, attention_mask_dev, label_scaled_distress_dev)

    # --- create dataloader ---
    # for empathy
    dataloader_emp_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_empathy_train, batch_size)
    dataloader_emp_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, batch_size)
    # for distress
    dataloader_dis_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_distress_train, batch_size)
    dataloader_dis_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, batch_size)


    # -------------------
    #  initialize model 
    # -------------------
    # source for creating and training model: [2] 
    #   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    # --- init model ---
    print('------------ initializing Model ------------')
    model = BertRegressor(bert_type=bert_type, train_only_bias=train_only_bias, activation_func=activation_func, dropout=dropout)
    model.to(device)

    # --- choose dataset and data loader based on empathy ---
    # per default use empathy label
    dataloader_train = dataloader_emp_train
    dataloader_dev = dataloader_emp_dev
    dataset_train = pytorch_dataset_emp_train
    dataset_dev = pytorch_dataset_emp_dev
    display_text = 'Using empathy data'
    if empathy_type == 'distress':
        dataloader_train = dataloader_dis_train  # needed for k fold
        dataloader_dev = dataloader_dis_dev  # needed for k fold
        dataset_train = pytorch_dataset_dis_train  # needed for k fold
        dataset_dev = pytorch_dataset_dis_dev  # needed for k fold
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')

    # --- optimizer ---
    # low learning rate to not get into catastrophic forgetting - Sun 2019
    # default epsilon by pytorch is 1e-8
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)

    # scheduler
    total_steps = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)

    # epochs
    loss_function = nn.MSELoss()
    
    if settings['kfold'] > 0:  # if kfold = 0, we ar enot doing kfold
        print('\n------------ Using kfold cross validation ------------\n')
        model, history = run_kfold(model, settings, dataset_dev, dataset_train, epochs, optimizer, scheduler, loss_function, device=device, k=settings['kfold'])
    else:
        model, history = model_utils.train_model(model, dataloader_train, dataloader_dev, epochs, optimizer, scheduler, loss_function, device=device, clip_value=2, use_scheduler=use_scheduler, use_early_stopping=use_early_stopping)
    
        # add model parameter size to history
        history['bert_param_size'] = np.zeros(history.shape[0]) + model.bert_parameter_count
        history['head_param_size'] = np.zeros(history.shape[0]) + model.head_parameter_count

    print(f"\nSave settings using model name: {settings['model_name']}\n")
    history.to_csv(root_folder + 'output/history_baseline_' + empathy_type + '_' + settings['model_name'] +  '.csv')
    
    if settings['save_model']:
        print(f"\nSave model using model name: {settings['model_name']}\n")
        torch.save(model.state_dict(), root_folder + 'output/model_baseline_' + empathy_type + '_' + settings['model_name'])
    print('Done')
    return model, history


def run_kfold(model, settings, dataset_dev, dataset_train, epochs, optimizer, scheduler, loss_function, device, k=5):
    model, history = model_utils.kfold_cross_val(model, settings, dataset_train, dataset_dev, optimizer, scheduler, loss_function, device, k=k, clip_value=2, early_stop_toleance=2, use_early_stopping=False, use_scheduler=False)
    # TODO, which model to save?
    return model, history

    

if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script

    settings = utils.arg_parsing_to_settings(args, early_stopping=False, learning_rate=2e-5, batch_size=16, bert_type='roberta-base', epochs=10, weight_decay=0.01, save_settings=True, use_scheduler=True, dropout=0.2, kfold=0)
    # ---- end function ----
    
    run(settings=settings)

