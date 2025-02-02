"""
Analysis and correlation of the moral direction by Schramowski et al. 2022 with the empathy dataset.
"""

import pickle
import matplotlib
from regex import F
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

import torch
from random import random
import decimal
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from EmpDim.funcs_mcm import BERTSentence
import utils.utils as utils
import utils.preprocessing as preprocessing
from utils.arguments import PCAArguments, DataTrainingArguments

from scipy.stats import pearsonr

# get imports from the submodule
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules/2022_Masterthesis_UnifiedPELT'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules/2022_Masterthesis_UnifiedPELT/transformers')) 
print(sys.path)
import importlib

unipelt_utils = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.utils')
unipelt_transformers = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers')
unipelt_preprocessing = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.preprocessing')
unipelt_arguments = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.arguments')
get_last_checkpoint = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'get_last_checkpoint')
is_main_process = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'is_main_process')
check_min_version = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.utils'), 'check_min_version')

AdapterConfig = getattr(unipelt_transformers, 'AdapterConfig')
AutoConfig = getattr(unipelt_transformers, 'AutoConfig')
AutoModelForSequenceClassification = getattr(unipelt_transformers, 'AutoModelForSequenceClassification')
AutoTokenizer = getattr(unipelt_transformers, 'AutoTokenizer')
DataCollatorWithPadding = getattr(unipelt_transformers, 'DataCollatorWithPadding')
EvalPrediction = getattr(unipelt_transformers, 'EvalPrediction')
HfArgumentParser = getattr(unipelt_transformers, 'HfArgumentParser')
MultiLingAdapterArguments = getattr(unipelt_transformers, 'MultiLingAdapterArguments')
PretrainedConfig = getattr(unipelt_transformers, 'PretrainedConfig')
Trainer = getattr(unipelt_transformers, 'Trainer')
TrainingArguments = getattr(unipelt_transformers, 'TrainingArguments')
default_data_collator = getattr(unipelt_transformers, 'default_data_collator')
set_seed = getattr(unipelt_transformers, 'set_seed')

COLORS = ['#029e72', '#e69f00', '#f0e441', '#57b4e8', '#6a329f']

def load_mort_pca(filename='../data/MoRT_projection/projection_model.p'):
    file = open(filename, 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()

    mort_pca = None
    try: 
        mort_pca = data['projection']
    except:
        print('No pca for MoRT found. PCA object will be None.')

    return mort_pca  # if not found, than pca will be None


# --- run on GPU if available ---
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("\n------------------ Using GPU. ------------------\n")
else:
    print("\n---------- No GPU available, using the CPU instead. ----------\n")
    device = torch.device("cpu")


sent_model = BERTSentence(device=device)

task_name = 'empathy'
data_args = DataTrainingArguments(task_name=task_name)
model_args = unipelt_arguments.ModelArguments()
training_args = TrainingArguments(output_dir='output/moral_output')

data_train_pd, data_dev_pd, data_test_pd = utils.load_data_complete(train_file=data_args.train_file, dev_file=data_args.validation_file, dev_label_file=data_args.validation_labels_file, test_file=data_args.test_file)
data_train_pd = utils.clean_raw_data(data_train_pd, keep_id=True)
data_dev_pd = utils.clean_raw_data(data_dev_pd, keep_id=True)
data_test_pd = utils.clean_raw_data(data_test_pd, keep_id=True)

# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    padding = False
    
# load tokenizer an dpreprocess data
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

dataset_emp_train, dataset_dis_train = preprocessing.get_preprocessed_dataset(data_train_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay', 'article_id'])
dataset_emp_dev, dataset_dis_dev = preprocessing.get_preprocessed_dataset(data_dev_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay', 'article_id'])
dataset_emp_test, dataset_dis_test = preprocessing.get_preprocessed_dataset(data_test_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay', 'article_id'])

# --- choose dataset and data loader based on empathy ---
# per default use empathy label
train_dataset = dataset_emp_train
eval_dataset = dataset_emp_dev
test_dataset = dataset_emp_test
display_text = 'Using empathy data'
if data_args.task_name == 'distress':
    train_dataset = dataset_dis_train
    eval_dataset = dataset_dis_dev
    test_dataset = dataset_dis_test
    display_text = "Using distress data"
print('\n------------ ' + display_text + ' ------------\n')

def remove_outliers(data_array):
    kept_idx, new_data = [], []
    avr = np.mean(data_array)
    std = np.std(data_array)
    is_outlier = lambda x: item > avr + std*2 or item < avr - std*2
    outlier_count = 0

    for idx, item in enumerate(data_array):
        if is_outlier(item):
            outlier_count += 1
            continue
        new_data.append(item)
        kept_idx.append(idx)
    kept_idx = np.array(kept_idx)
    new_data = np.array(new_data)
    print(f'Detected {outlier_count} outlier(s)')
    return new_data, kept_idx


def scatter_moral_empdis(pca_features, labels):

    pca_dim = pca_features.shape[1]

    for i in range(pca_dim):
        moral_dim_pc_i = pca_features[:, i]
        moral_dim_no_outlier_pc_i, kept_idx = remove_outliers(moral_dim_pc_i)
        labels_i = np.array(labels)[kept_idx]
        r, p = pearsonr(moral_dim_no_outlier_pc_i, labels_i)
        r = r[0]
        plt.scatter(labels_i, moral_dim_no_outlier_pc_i)
        plt.ylabel('MoRT score')
        plt.xlabel(f'{data_args.task_name} score')
        plt.title(f'Scatter plots MoRT: PC {i+1}. pearson r: {r:.4f}.')
        plt.savefig(get_output_dir() + f'/scatter_moral_{data_args.task_name}_{i+1}.pdf')
        plt.close()
        try:
            new_row = pd.DataFrame().from_dict({'pearson_r':[r], 'pearson_p': [p], 'princ_comp':[(i+1)], 'note':['MD ess - label (Without outliers)'], 'task_name': [data_args.task_name]})
            correlations_pd = pd.concat([correlations_pd, new_row], ignore_index=True)
        except:
            pass

def plot_moral_empdis(bins, binned_data):
    binned_ave, binned_std, final_bins = [], [], []
    for idx, bin in enumerate(binned_data):
        if len(bin) >= 1:
            ave_bin = np.mean(bin, axis=0)
            std_bin = np.std(bin, axis=0)
            binned_ave.append(ave_bin)
            binned_std.append(std_bin)
            final_bins.append(bins[idx])
            
    binned_ave = np.array(binned_ave)
    binned_std = np.array(binned_std)

    lower_std_bound = binned_ave - binned_std
    upper_std_bound = binned_ave + binned_std
    for i in range(binned_ave.shape[1]):
        plt.plot(final_bins, binned_ave[:, i], c=COLORS[i], label=f'PC {i+1}')
        plt.fill_between(final_bins, lower_std_bound[:, i], upper_std_bound[:, i], color=COLORS[i], alpha=0.5)
    plt.title('The average moral score with the std')
    plt.ylabel('Average moral score')
    plt.xlabel(f'{data_args.task_name} score (in bins)')
    plt.legend()
    plt.savefig(get_output_dir() + f'/ave_moral_{data_args.task_name}.pdf')
    return binned_ave, binned_std, final_bins


def bin_data(labels, moral_pca, bin_size=0.1):
    # - create bins -
    decimal_count = abs(decimal.Decimal(str(bin_size)).as_tuple().exponent)
    min_score = min([item for item in labels])
    max_score = max([item for item in labels])
    bins_start = math.floor(min_score * (10**decimal_count)) / (10**decimal_count)
    bins_end = math.ceil(max_score * (10**decimal_count)) / (10**decimal_count)
    # add the end point to the bins as well, to get the upper range for the elements
    # this will be removed later on, since it is not actually a bin
    bins = np.arange(bins_start, bins_end + bin_size, bin_size)

    # - divide data into bins - 
    binned_pca = [[] for i in range(len(bins))]
    binned_labels = [[] for i in range(len(bins))]
    for idx, score in enumerate(labels):
        min_idx = np.where(bins <= score)[0]
        max_idx = np.where(bins > score)[0] - 1
        if len(max_idx) == 0 and len(min_idx) == len(bins): # the score goes into the last bin
            max_idx = np.array([len(bins) - 1])  # put last index in here
        
        item_bin_idx = np.intersect1d(min_idx, max_idx)

        if len(item_bin_idx) > 0:
            item_bin_idx = item_bin_idx[0]
            moral_pca_i = moral_pca[idx]
            binned_pca[item_bin_idx].append(moral_pca_i)
            binned_labels[item_bin_idx].append(score)
    # remove last bin, because it is 0 anyways, just needed it for the calculation
    binned_pca = binned_pca[:-1]
    binned_labels = binned_labels[:-1]
    bins = bins[:-1]

    return binned_pca, binned_labels, bins


def get_output_dir():
    output_name = training_args.output_dir
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    return output_name

# ------------------
# create moral score
# ------------------
essays = train_dataset['essay']
labels = np.array(train_dataset['label'])

essay_embeddings = sent_model.get_sen_embedding(essays)

mort_pca = load_mort_pca(filename=data_args.data_dir + '/MoRT_projection/projection_model.p')
moral_dim = mort_pca.transform(essay_embeddings)


try:
    print(moral_dim.shape)
except:
    pass

pca_dim = moral_dim.shape[1]


csv_path = get_output_dir() + '/moral_correlations.csv'
if not os.path.exists(csv_path):
    correlations_pd = pd.DataFrame()
else: 
    correlations_pd = pd.read_csv(csv_path, index_col=0)


for i in range(pca_dim):
    print(f'correlation of PC {i+1}')
    moral_dim_pc_i = moral_dim[:, i]
    print('labels.shape', labels.shape)
    print('moral_dim_pc_i.shape', moral_dim_pc_i.shape)
    r, p = pearsonr(moral_dim_pc_i, labels)
    r = r[0]
    print(f'r: {r}, p: {p}')
    new_row = pd.DataFrame().from_dict({'pearson_r':[r], 'pearson_p': [p], 'princ_comp':[(i+1)], 'note':['MD ess - label (With outliers)'], 'task_name': [data_args.task_name]})
    correlations_pd = pd.concat([correlations_pd, new_row], ignore_index=True)

scatter_moral_empdis(moral_dim, labels)

binned_pca, binned_labels, bins = bin_data(labels, moral_dim, 0.1)
plot_moral_empdis(bins, binned_pca)

# -------------------------
# Do analysis with articles
# -------------------------

articles = utils.load_articles(data_root_folder=data_args.data_dir)
# lower textual data
articles['text'] = articles['text'].apply(lambda x: x.lower())
articles = preprocessing.tokenize_data(articles, 'text')  # creates column text_tok
articles = preprocessing.lemmatize_data(articles, 'text_tok')

# --- Generate MoRT for articles ---
articles_text = articles['text']
article_ids = articles['article_id']

articles_embeddings = sent_model.get_sen_embedding(articles_text)

#mort_pca = load_mort_pca(filename=data_args.data_dir + '/MoRT_projection/projection_model.p')
moral_dim_articles = mort_pca.transform(articles_embeddings)  # dim = 5

essay_article_ids = np.array(train_dataset['article_id'])

for i in range(moral_dim_articles.shape[1]):
    articles_mort_i = moral_dim_articles[:, i]
    article_ids_list = list(article_ids)
    indices = [article_ids_list.index(id) for id in list(essay_article_ids)]
    article_mort_per_essay = np.take(articles_mort_i, indices)

    moral_dim_pc_i = moral_dim[:, i]
    r, p = pearsonr(article_mort_per_essay, labels)
    if isinstance(r, list): r = r[0]
    print(f'Article MoRT and labels. r: {r}, p: {p}')
    new_row = pd.DataFrame().from_dict({'pearson_r':[r], 'pearson_p': [p], 'princ_comp':[(i+1)], 'note':['MoRT_art - labels'], 'task_name': [data_args.task_name]})
    correlations_pd = pd.concat([correlations_pd, new_row], ignore_index=True)

    r, p = pearsonr(article_mort_per_essay, moral_dim_pc_i)
    if isinstance(r, list): r = r[0]
    print(f'Article MoRT and essay MoRT. r: {r}, p: {p}')
    new_row = pd.DataFrame().from_dict({'pearson_r':[r], 'pearson_p': [p], 'princ_comp':[(i+1)], 'note':['MoRT_art - MoRT_essay'], 'task_name': [data_args.task_name]})
    correlations_pd = pd.concat([correlations_pd, new_row], ignore_index=True)

    # Does their difference correlate with the empathy score?
    #rmse = np.sqrt((article_mort_per_essay**2 + moral_dim_pc_i**2) / 2) # root mean squared error
    #np.sqrt((a - b)**2)
    similarity_morts = np.sqrt((article_mort_per_essay - moral_dim_pc_i)**2)
    r, p = pearsonr(similarity_morts, labels)
    if isinstance(r, list): r = r[0]
    print(f'Sim(Article MoRT, essay MoRT) and labels. r: {r}, p: {p}')
    new_row = pd.DataFrame().from_dict({'pearson_r':[r], 'pearson_p': [p], 'princ_comp':[(i+1)], 'note':['Sim(MoRT_art, MoRT_essay) - labels'], 'task_name': [data_args.task_name]})
    correlations_pd = pd.concat([correlations_pd, new_row], ignore_index=True)

correlations_pd.to_csv(csv_path)
