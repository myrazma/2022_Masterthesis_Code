# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
# [3] Adapter versions config:
#   https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
# [4] Unify parameter efficient training
#   https://github.com/jxhe/unify-parameter-efficient-tuning
#
# ------------------------------

# utils
from logging import root
import time
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
# Transformers, torch and model utils
from transformers import AutoTokenizer, BertModel, BertConfig, BertForSequenceClassification, AutoModel, RobertaModel
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, RobertaTokenizer
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import get_linear_schedule_with_warmup
import transformers.adapters as adapters
from transformers.adapters import AutoAdapterModel, RobertaAdapterModel, PredictionHead
from transformers.adapters import MAMConfig, AdapterConfig, PrefixTuningConfig, ParallelConfig, HoulsbyConfig
from transformers.adapters import configuration as adapter_configs
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torch.optim import AdamW
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

# TODO: I need to structure for adapters
# TODO: Use Trainer / Adaptertrainer
class RegressionModelAdapters(nn.Module):
    def __init__(self, bert_type, task_type, adapter_config, activation_func='relu', dropout=0.5):
        super(RegressionModelAdapters, self).__init__()
        D_in, D_out = 768, 1 
        
        adapter_name = task_type + '_adapter'
        self.bert = RobertaAdapterModel.from_pretrained(bert_type)

        # Enable adapter training
        # task adapter - only add if not existing
        if adapter_name not in self.bert.config.adapters:
            print('adding adapter')
            self.bert.add_adapter(adapter_name, config=adapter_config, set_active=True)
        #self.bert.set_active_adapters(adapter_name)
        self.bert.train_adapter(adapter_name)  # set adapter into training mode and freeze parameters in the transformer model
        
        # print frozen parameters
        if False:
            names = [n for n, p in self.bert.named_parameters()]
            paramsis = [param for param in self.bert.parameters()]
            for n, p in zip(names, paramsis):
                print(f"{n}: {p.requires_grad}")
        
        self.regression_head = model_utils.RegressionHead(D_in=D_in, D_out=D_out, activation_func=activation_func, dropout=dropout)

        self.bert_parameter_count = model_utils.count_updated_parameters(self.bert.parameters())
        self.head_parameter_count = model_utils.count_updated_parameters(self.regression_head.parameters())
        

    def forward(self, input_ids, attention_masks):
        bert_outputs = self.bert(input_ids, attention_masks)
        outputs = self.regression_head(bert_outputs)
        return outputs


class MyDataset(PyTorchDataset):
    def __init__(self, input_ids, attention_mask, labels, device):
        """Initializer for SeqDataset
        Args:
            seq (np.array) [x, y]: The sequences of shape (sample_size, max_seq_size)
            labels (np.array) [x, y]: The labels of shape (sample_size, max_seq_size)
        """
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor)
        self.input_ids = torch.from_numpy(input_ids).type(torch.FloatTensor)
        self.attention_mask = torch.from_numpy(attention_mask).type(torch.LongTensor)

    def __len__(self):
        """Implement len function of type Dataset
        Returns:
            int: The length of the dataset
        """
        return len(self.labels)
            
    def __getitem__(self, idx):
        """Implement get_item function of type Dataset
        Args:
            idx (int): The index of the item to get
        Returns:
            tensor [y], tensor [y]: The sequence, The labels
        """
        item = {}
        item['attention_masks'] = self.attention_mask[idx].int()
        item['input_ids'] = self.input_ids[idx].int()
        item['label'] = self.labels[idx].float()
        return item


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

    
def tokenize(batch, tokenizer, column):
    # Source: [1] - https://huggingface.co/docs/transformers/training
    # longest is around 200
    return tokenizer(batch[column], padding='max_length', truncation=True, max_length=256)


def create_dataloaders(inputs, masks, labels, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_dataloaders_multi_in(inputs, masks, labels, lexical_features, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    lexical_features_tensor = torch.tensor(lexical_features)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, lexical_features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_adapter_config(config_name, print_config=True):
    """available adapters from adapter hub (18.04.22):
    (They can be used passing a string)
    
    ADAPTER_CONFIG_MAP = {
    "pfeiffer": PfeifferConfig(),  # Pfeiffer2020
    "houlsby": HoulsbyConfig(),
    "pfeiffer+inv": PfeifferInvConfig(),
    "houlsby+inv": HoulsbyInvConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "compacter": CompacterConfig(),
    "prefix_tuning": PrefixTuningConfig(),  # Li and Liang (2021)
    "prefix_tuning_flat": PrefixTuningConfig(flat=True),
    "parallel": ParallelConfig(),  # He2021
    "scaled_parallel": ParallelConfig(scaling="learned"),
    "mam": MAMConfig(),  # He2021
    }
    
    mam config is the same as according to [3]
    config = ConfigUnion(
        PrefixTuningConfig(bottleneck_size=800),
        ParallelConfig(),
        )
    """

    # load the predefined adapter configurations from the hub
    configs_dict = copy.deepcopy(adapter_configs.ADAPTER_CONFIG_MAP)

    # create own config options using some configs from here: https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
    myprefixtuning_config = PrefixTuningConfig(flat=False, prefix_length=30, bottleneck_size=200)
    configs_dict['myprefixtuning'] = myprefixtuning_config

    # adapted from from He2021; with reduction factor = 1.5 it comes closer to 512 dim than with reduction factor of 2 (from 768 to 384)
    # however, it cant be an integer, so we need to leave it like that
    mymam_config = MAMConfig(PrefixTuningConfig(bottleneck_size=30), ParallelConfig(reduction_factor=2))
    configs_dict['mymam'] = mymam_config
    
    # select config
    if config_name in configs_dict.keys():
        config = configs_dict[config_name]
    else:
        print(f'\nMyWarning: Could not find an adapter configuration for {config_name}. Please select one of the following:\n {configs_dict.keys()}\n')
        sys.exit(-1)
    if print_config: print(config)
    """
    Source: Flexible configurations with ConfigUnion 
    https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
    
    from transformers.adapters import AdapterConfig, ConfigUnion

    config = ConfigUnion(
        AdapterConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
        AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
    )
    model.add_adapter("union_adapter", config=config)
    """
    return config


def run(settings, root_folder=""):

    data_root_folder = root_folder + 'data/'
    output_root_folder = root_folder + 'output/'

    #logging.set_verbosity_warning()
    #logging.set_verbosity_error()
    use_gpu = False
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
        use_gpu = True
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")
        use_gpu = False
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
    adapter_type = settings['adapter_type']
    use_early_stopping = settings['early_stopping']
    weight_decay = settings['weight_decay']
    use_scheduler = settings['scheduler']
    activation_func = settings['activation']
    dropout = settings['dropout']

    adapter_config = get_adapter_config(adapter_type)
    # -------------------
    #   load data
    # -------------------
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)

    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)

    # save raw essay (will not be tokenized by BERT)
    data_train_pd['essay_raw'] = data_train_pd['essay']
    data_dev_pd['essay_raw'] = data_dev_pd['essay']
    
    # tokenize them already and create column essay_raw_tok
    data_train_pd = preprocessing.tokenize_data(data_train_pd, 'essay_raw')
    data_dev_pd = preprocessing.tokenize_data(data_dev_pd, 'essay_raw')
    
    # create lexical features
    fc = preprocessing.FeatureCreator(data_root_folder=data_root_folder)
    data_train_pd = fc.create_lexical_feature(data_train_pd, 'essay_raw_tok')
    data_dev_pd = fc.create_lexical_feature(data_dev_pd, 'essay_raw_tok')

    # --- Create hugginface datasets ---
    # TODO: Use all data later on
    data_train = pd_to_dataset(data_train_pd)
    data_dev = pd_to_dataset(data_dev_pd)

    #  Create hugginface datasetsdict
    # data_train_dev = pd_to_datasetdict(data_train, data_dev)
   
    # -------------------
    #   preprocess data
    # -------------------

    # --- tokenize data ---
    tokenizer = RobertaTokenizer.from_pretrained(bert_type)

    data_train_encoded = data_train.map(lambda x: tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)


    # --- shuffle data ---
    data_train_encoded_shuff = data_train_encoded.shuffle(seed=my_seed)
    data_dev_encoded_shuff = data_dev_encoded.shuffle(seed=my_seed)

    # get input_ids, attention_mask and labels
    # train
    input_ids_train = np.array(data_train_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_train_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_train = np.array(data_train_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_train = np.array(data_train_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    lexical_emp_train = np.array(data_train_encoded_shuff["empathy_word_rating"]).astype(np.float32).reshape(-1, 1)
    lexical_dis_train = np.array(data_train_encoded_shuff["distress_word_rating"]).astype(np.float32).reshape(-1, 1)
    
    # dev
    input_ids_dev = np.array(data_dev_encoded_shuff["input_ids"]).astype(int)
    attention_mask_dev = np.array(data_dev_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_dev = np.array(data_dev_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_dev = np.array(data_dev_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    lexical_emp_dev = np.array(data_dev_encoded_shuff["empathy_word_rating"]).astype(np.float32).reshape(-1, 1)
    lexical_dis_dev = np.array(data_dev_encoded_shuff["distress_word_rating"]).astype(np.float32).reshape(-1, 1)

    # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
    label_scaled_empathy_train = preprocessing.normalize_scores(label_empathy_train, (1,7))
    label_scaled_empathy_dev = preprocessing.normalize_scores(label_empathy_dev, (1,7))
    label_scaled_distress_train = preprocessing.normalize_scores(label_distress_train, (1,7))
    label_scaled_distress_dev = preprocessing.normalize_scores(label_distress_dev, (1,7))

    # --- create dataloader ---
    # with lexical data
    # for empathy
    #dataloader_emp_train = create_dataloaders_multi_in(input_ids_train, attention_mask_train, label_scaled_empathy_train, lexical_emp_train, batch_size)
    #dataloader_emp_dev = create_dataloaders_multi_in(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, lexical_emp_dev, batch_size)
    # for distress
    #dataloader_dis_train = create_dataloaders_multi_in(input_ids_train, attention_mask_train, label_scaled_distress_train, lexical_dis_train, batch_size)
    #dataloader_dis_dev = create_dataloaders_multi_in(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, lexical_dis_dev, batch_size)

    # without lexical data
    # for empathy
    dataloader_emp_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_empathy_train, batch_size)
    dataloader_emp_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, batch_size)
    # for distress
    dataloader_dis_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_distress_train, batch_size)
    dataloader_dis_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, batch_size)

    # --- choose dataset ---
    # per default use empathy label
    dataloader_train = dataloader_emp_train
    dataloader_dev = dataloader_emp_dev
    display_text = 'Using empathy data'
    if empathy_type == 'distress':
        dataloader_train = dataloader_dis_train
        dataloader_dev = dataloader_dis_dev
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')


    # --- init model ---
    print('------------ initializing Model ------------')
    model = RegressionModelAdapters(bert_type=bert_type,task_type=empathy_type, adapter_config=adapter_config, activation_func=activation_func, dropout=dropout)
    model.to(device)
    print(model)

    # -------------------------------
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
   
    model, history = model_utils.train_model(model, dataloader_train, dataloader_dev, epochs, optimizer, scheduler, loss_function, device, clip_value=2, use_early_stopping=use_early_stopping, use_scheduler=use_scheduler)
    
    # add model parameter size to history
    history['bert_param_size'] = np.zeros(history.shape[0]) + model.bert_parameter_count
    history['head_param_size'] = np.zeros(history.shape[0]) + model.head_parameter_count

    print(f"\nSave settings using model name: {settings['model_name']}\n")
    history.to_csv(output_root_folder + 'history_adapters_' + empathy_type + '_' + settings['model_name'] +  '.csv')
        
    if settings['save_model']:
        print(f"\nSave model using model name: {settings['model_name']}\n")
        torch.save(model.state_dict(), output_root_folder + 'model_adapters_' + empathy_type + '_' + settings['model_name'])
    print('Done')
    return model, history


if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script

    settings = utils.arg_parsing_to_settings(args, learning_rate=2e-5, batch_size=16, epochs=10, save_settings=True, bert_type='roberta-base', weight_decay=0.01, save_settings=True)
    # ---- end function ----
    
    run(settings=settings)

