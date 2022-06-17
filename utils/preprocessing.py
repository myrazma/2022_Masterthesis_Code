
import pandas as pd
import numpy as np
import sys
import json
from collections import defaultdict
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, DatasetDict
#nltk.download('wordnet')
#nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

# Load english stop words
STOPWORDS_EN = set(stopwords.words('english'))


# ---------------------------------------------------
#               Preprocessing methods
# ---------------------------------------------------
def remove_non_alpha(text_tok):
    return [word for word in text_tok if word.isalpha()]


def tokenize_data(data, column):
    """ Tokenize text in data based on space and punctuation using nltk word_tokenizer
    created new column wiht column + suffix: '_tok'

    Args:
        data (pd.DataFrame): The data including the texts
        column (str): The name of the column holding the text

    Returns:
        pd.DataFrame: The dataframe with the tokenized texts (suffix: '_tok')
    """
    data[column + '_tok'] = data[column].apply(lambda x: nltk.word_tokenize(x))
    return data

def tokenize_single_text(text):
    """ Tokenize text in data based on space and punctuation using nltk word_tokenizer

    Args:
        data (pd.DataFrame): The data including the texts
        column (str): The name of the column holding the text

    Returns:
        pd.DataFrame: The dataframe with the tokenized texts (suffix: '_tok')
    """
    return nltk.word_tokenize(text)


def lemmatize_data(data, column):
    """Lemmatize tokenized textual data using WordNetLemmatizer from nltk

    Args:
        data (pd.DataFrame): The data including the tokenized version of the texts
        column (str): The name of the column holding the tokenized text

    Returns:
        pd.DataFrame: The dataframe with a new column (suffix: '_lem')
    """
    lemmatizer = WordNetLemmatizer()  # lemmatize
    data[column + '_lem'] = data[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return data


def remove_stopwords(text_tok):
    """Remove (english) stopwords from a tokenized texts

    Args:
        text_tok (list(str)): Tokenized text

    Returns:
        list(str): The tokenized text without stopwords
    """
    text_processed = [word for word in text_tok if not word.lower() in STOPWORDS_EN]
    return text_processed


def normalize_scores(data, input_interval):
    """Maps from desired input intervall to [0,1]

    Args:
        data (np.array): The data
        input_interval ((int,int)): _description_

    Returns:
        _type_: _description_
    """
    normalized = (data - input_interval[0]) / (input_interval[1] - input_interval[0])
    return normalized


def tokenize(batch, tokenizer, column, max_length=256):
    # Source: [1] - https://huggingface.co/docs/transformers/training
    # longest is around 200
    return tokenizer(batch[column], padding='max_length', truncation=True, max_length=max_length)


# ---------------------------------------------------
#               Data set methods
# ---------------------------------------------------

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


def pd_to_dataset(data_df):
    """Create hugginface dataset from pandas dataframe

   Args:
        data_df (pd.DataFrame): _description_

    Returns:
        Dataset: The huggingface dataset from datasets.Dataset
    """
    data_df = Dataset.from_pandas(data_df)
    return data_df


def create_dataloaders_multi_in(inputs, masks, labels, lexical_features, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    lexical_features_tensor = torch.tensor(lexical_features)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, lexical_features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ---------------------------------------------------
#         Complete Preprocessing of the data
# ---------------------------------------------------


def get_preprocessed_dataset(data_pd, tokenizer, seed, return_huggingface_ds=False, padding='max_length'):
    """Preprocess the data from input as pandas pd and return a TensorDataset
    
    Do the following steps:
    1. Tokenize data using the tokeniezr
    2. Shuffle the data
    3. Normalize the empathy and distress scores from [1,7] to [0,1]
    4. Create Tensor Dataset (or huggingface dataset)

    Returns the datasets separated by empatyh and distress and dev and train.

    Args:
        data_pd (pd.DataFrame): _description_
        tokenizer (_type_): _description_
        seed (int): The seed
        return_huggingface_ds (bool): Return data as huggingface dataset from datasets library. Default: False.

    Returns:
        TensorDataset, TensorDataset, TensorDataset, TensorDataset: dataset_emp_train, dataset_emp_dev, dataset_dis_train, dataset_dis_dev
    """

    # check if the dataset has labels (not True for test set)
    if 'empathy' in data_pd.columns or 'distress' in data_pd.columns:
        has_label = True
    else:
        has_label = False

    # --- Create hugginface datasets ---
    data = pd_to_dataset(data_pd)

    # -------------------
    #   preprocess data
    # -------------------
    data_encoded = data.map(lambda x: tokenize(x, tokenizer, 'essay', padding), batched=True, batch_size=None)


    # --- shuffle data ---
    data_encoded_shuff = data_encoded.shuffle(seed=seed)
    # get input_ids, attention_mask and labels as numpy arrays and cast types
    input_ids_train = np.array(data_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_encoded_shuff["attention_mask"]).astype(int)
    if has_label:
        label_empathy_train = np.array(data_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
        label_distress_train = np.array(data_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)

        # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
        label_scaled_empathy_train = normalize_scores(label_empathy_train, (1,7))
        label_scaled_distress_train = normalize_scores(label_distress_train, (1,7))

        # --- create datasets ---
        # for empathy
        dataset_emp_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_empathy_train)
        # for distress
        dataset_dis_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_distress_train)

        if return_huggingface_ds:
            # --- create panda DataFrame datasets ---
            # for empathy
            dataset_emp_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train, 'label': label_scaled_empathy_train})
            # for distress
            dataset_dis_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train, 'label': label_scaled_distress_train})
    else:  # for test set
        # --- create datasets ---
        dataset_emp_train = create_tensor_data(input_ids_train, attention_mask_train)
        dataset_dis_train = create_tensor_data(input_ids_train, attention_mask_train)

        if return_huggingface_ds:
            # --- create panda DataFrame datasets ---
            dataset_emp_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train})
            dataset_dis_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train})

    return dataset_emp_train, dataset_dis_train



def get_preprocessed_dataset_huggingface(data_pd, tokenizer, seed, return_huggingface_ds=True, padding='max_length', max_length=256):
    """Preprocess the data from input as pandas pd and return a TensorDataset
    
    Do the following steps:
    1. Tokenize data using the tokeniezr
    2. Shuffle the data
    3. Normalize the empathy and distress scores from [1,7] to [0,1]
    4. Create Tensor Dataset (or huggingface dataset)
    Returns the datasets separated by empatyh and distress and dev and train.
    Args:
        data_pd (pd.DataFrame): _description_
        tokenizer (_type_): _description_
        seed (int): The seed
        return_huggingface_ds (bool): Return data as huggingface dataset from datasets library. Default: False.
    Returns:
        TensorDataset, TensorDataset, TensorDataset, TensorDataset: dataset_emp_train, dataset_emp_dev, dataset_dis_train, dataset_dis_dev
    """

    # check if the dataset has labels (not True for test set)
    if 'empathy' in data_pd.columns or 'distress' in data_pd.columns:
        has_label = True
    else:
        has_label = False

    # --- Create hugginface datasets ---
    data = pd_to_dataset(data_pd)

    # -------------------
    #   preprocess data
    # -------------------
    data_encoded = data.map(lambda x: tokenize(x, tokenizer, 'essay', max_length=max_length), batched=True, batch_size=None)


    # --- shuffle data ---
    data_encoded_shuff = data_encoded.shuffle(seed=seed)
    # get input_ids, attention_mask and labels as numpy arrays and cast types
    input_ids_train = np.array(data_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_encoded_shuff["attention_mask"]).astype(int)
    if has_label:
        label_empathy_train = np.array(data_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
        label_distress_train = np.array(data_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)

        # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
        label_scaled_empathy_train = normalize_scores(label_empathy_train, (1,7))
        label_scaled_distress_train = normalize_scores(label_distress_train, (1,7))

        # --- create datasets ---
        # for empathy
        dataset_emp_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_empathy_train)
        # for distress
        dataset_dis_train = create_tensor_data(input_ids_train, attention_mask_train, label_scaled_distress_train)

        if return_huggingface_ds:
            # --- create panda DataFrame datasets ---
            # for empathy
            dataset_emp_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train, 'label': label_scaled_empathy_train})
            # for distress
            dataset_dis_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train, 'label': label_scaled_distress_train})
    else:  # for test set
        # --- create datasets ---
        dataset_emp_train = create_tensor_data(input_ids_train, attention_mask_train)
        dataset_dis_train = create_tensor_data(input_ids_train, attention_mask_train)

        if return_huggingface_ds:
            # --- create panda DataFrame datasets ---
            dataset_emp_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train})
            dataset_dis_train = Dataset.from_dict({'input_ids': input_ids_train, 'attention_mask':attention_mask_train})

    return dataset_emp_train, dataset_dis_train