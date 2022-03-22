from transformers import BertModel, BertConfig, BertForSequenceClassification
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertTokenizer
import torch

# import own module
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)
import utils


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

    
def tokenize(batch, tokenizer):
    return tokenizer(batch['essay'], padding=True, truncation=True, max_length=512)


def run():
    # load data
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder="data/")
    print(data_train_pd.shape)
    data_train_pd = utils.preprocessing(data_train_pd)
    data_dev_pd = utils.preprocessing(data_dev_pd)
    
    #  Create hugginface datasets
    data_train = pd_to_dataset(data_train_pd)
    data_dev = pd_to_dataset(data_dev_pd)

    #  Create hugginface datasetsdict
    # data_train_dev = pd_to_datasetdict(data_train, data_dev)


    bert_type = "bert-base-uncased"
    
    # Tokenize essays
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    data_train_encoded = data_train.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)

    print(data_train_encoded.column_names)

    model = BertForSequenceClassification.from_pretrained(bert_type)


    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()

    # Initializing a model from the bert-base-uncased style configuration
    model = BertModel(configuration)

    # Accessing the model configuration
    configuration = model.config
    print('Done')


if __name__ == '__main__':
    run()

