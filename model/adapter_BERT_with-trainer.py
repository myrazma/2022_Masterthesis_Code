# ---------- Sources ----------
# Training adaper with roberta and AdapterTrainer
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
from transformers.adapters import AutoAdapterModel, RobertaAdapterModel, PredictionHead, MAMConfig
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    def __init__(self, bert_type, task_type, drop_rate=0.2, adapter_config="pfeiffer"):
        super(RegressionModelAdapters, self).__init__()
        D_in = 768
        Bert_out = 100
        Multi_in = Bert_out + 1
        Hidden_Regressor = 50
        D_out = 1

        #model.add_adapter("mam_adapter", config=config)
        head_name = task_type + '_head'
        adapter_name = task_type + '_adapter'
        self.bert = RobertaAdapterModel.from_pretrained(bert_type)
        # TODO
        #regr_head = RegressionHead(self.bert, head_name)
        #self.bert.register_custom_head(head_name, regr_head)
        #self.bert.add_custom_head(head_type=head_name, head_name='_' + head_name)
        self.bert.add_classification_head(adapter_name, num_labels=1)  # if label is one, regression is used
        self.bert.add_adapter(adapter_name, config=adapter_config)
        self.bert.set_active_adapters(adapter_name)
        self.bert.train_adapter(adapter_name)  # set adapter into training mode and freeze parameters in the transformer model

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        return outputs


    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        #cls_output = kwargs.pop("pooled_output")
        #print(type(outputs))
        print('-----------------------')
        print(cls_output)
        print(attention_mask)
        print(attention_mask)
        
        print(type(cls_output))
        #if cls_output is None:
        cls_output = outputs[0][:, 0]
        print(cls_output)
        logits = super().forward(cls_output)
        loss = None
        labels = kwargs.pop("labels", None)
        print(labels)
        print('-----------------------')
        #print(type(attention_mask))
        #outputs = self.bert(input_ids, attention_masks)
        bert_output = outputs[1]

        # concat bert output with multi iput - lexical data
        after_bert_outputs = self.bert_head(bert_output)
    
        # combine bert output (after short ffn) with lexical features
        #concat = torch.cat((after_bert_outputs, lexical_features), 1)
        #outputs = self.regressor(concat)
        return after_bert_outputs


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


def score_correlation(y_pred, y_true):
    """Correlate prediciton and true value using pearson r

    Args:
        y_pred (array): The predicted labels
        y_true (array): The true labels

    Returns:
        r, p (float, float): pearson r, p-value
    """
    r, p = pearsonr(y_true, y_pred)
    return r, p
    

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
    adapter_config = settings['adapter_type']
    weight_decay = settings['weight_decay']


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
    data_train_pd = fc.create_lexical_feature_dataframe(data_train_pd, 'essay_raw_tok')
    data_dev_pd = fc.create_lexical_feature_dataframe(data_dev_pd, 'essay_raw_tok')

    # --- Create hugginface datasets ---
    # TODO: Use all data later on
    data_train = pd_to_dataset(data_train_pd)
    data_dev = pd_to_dataset(data_dev_pd)

    #  Create hugginface datasetsdict
    # data_train_dev = pd_to_datasetdict(data_train, data_dev)
    print(data_train)
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

    # -------------------
    #  initialize pre trained model an 
    # -------------------
    # source for creating and training model: [2] 
    #   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    # --- load or pre-train bert here ---
    #pre_trained_emp_bert = BertRegressor()
    #pre_trained_emp_bert = torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330', map_location=torch.device('cpu'))
    #if use_gpu:
    #    pre_trained_emp_bert.load_state_dict(torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330'))
    #else:
    #    pre_trained_emp_bert.load_state_dict(torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330',map_location=torch.device('cpu')))
    # get output from pre-trained bert
    #bert_outputs_emp_train = pre_trained_emp_bert(torch.tensor(input_ids_train), torch.tensor(attention_mask_train))
    #bert_outputs_emp_dev = pre_trained_emp_bert(torch.tensor(input_ids_dev), torch.tensor(attention_mask_dev))

    #emp_dev_corr, _ = score_correlation(np.array(bert_outputs_emp_dev), np.array(label_empathy_dev))
    #print(emp_dev_corr)
    #return
    # -- pearson correlation --


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
    
    #dataloader_emp_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_empathy_train, batch_size)
    #dataloader_emp_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, batch_size)
    # for distress
    #dataloader_dis_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_distress_train, batch_size)
    #dataloader_dis_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, batch_size)
    
    pytorch_dataset_emp_train = model_utils.MyDataset(input_ids_train, attention_mask_train, label_scaled_empathy_train, device)
    pytorch_dataset_emp_dev = model_utils.MyDataset(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, device)
    pytorch_dataset_dis_train = model_utils.MyDataset(input_ids_train, attention_mask_train, label_scaled_distress_train, device)
    pytorch_dataset_dis_dev = model_utils.MyDataset(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, device)
  
    # --- choose dataset ---
    # per default use empathy label
    dataset_train = pytorch_dataset_emp_train
    dataset_dev = pytorch_dataset_emp_dev
    display_text = 'Using empathy data'
    if empathy_type == 'distress':
        dataset_train = pytorch_dataset_dis_train
        dataset_dev = pytorch_dataset_dis_dev
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')


    # --- init model ---
    print('------------ initializing Model ------------')
   
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=1,
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )
    #model = RobertaAdapterModel.from_pretrained(bert_type)
    # TODO
    head_name = empathy_type + '_head'
    adapter_name = empathy_type + '_adapter'
    model.add_classification_head(adapter_name, num_labels=1, activation_function='relu')  # if label is one, regression is used
    model.add_adapter(adapter_name, config=adapter_config)
    model.set_active_adapters(adapter_name)
    model.train_adapter(adapter_name)  # set adapter into training mode and freeze parameters in the transformer model
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)


    # -------------------------------
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=200,
        output_dir="./output",
        overwrite_output_dir=False,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        weight_decay=weight_decay
    )

    def compute_correlation(p: EvalPrediction):
        r, p = pearsonr(p.label_ids, p.predictions)
        return {"corr": r[0]}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        compute_metrics=compute_correlation,
    )

    history = trainer.train()
    print(history)
    metrics = trainer.evaluate()
    print(metrics)
    return



if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script

    settings = utils.arg_parsing_to_settings(args, default_learning_rate=5e-5, default_batch_size=16, default_adapter_type='parallel', default_epochs=10)
    # ---- end function ----
    
    run(settings=settings)

