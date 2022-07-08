import pickle
import matplotlib
from regex import F
import sklearn
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

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
#import transformers

unipelt_utils = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.utils')
#try:
unipelt_transformers = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers')
unipelt_preprocessing = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.preprocessing')
unipelt_arguments = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.arguments')
#except:
#    print('The UniPelt Input is not available. Probably an import in "submodules.2022_Masterthesis_UnifiedPELT" not working.')
#    sys.exit(-1)
# use importlib in this case, because of the name in submodule starting with a number
get_last_checkpoint = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'get_last_checkpoint')
is_main_process = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.trainer_utils'), 'is_main_process')
check_min_version = getattr(importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers.utils'), 'check_min_version')

#from transformers.trainer_utils import get_last_checkpoint, is_main_process
#from transformers.utils import check_min_version
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

COLORS = ['#029e72', '#e69f00', '#f0e441', '#57b4e8']

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

# only use cpu for this analysis when running with a100
#device = 'cpu'

sent_model = BERTSentence(device=device)

task_name = 'distress'
# TODO, add mort file to DataTrainArgs
data_args = DataTrainingArguments(task_name=task_name)
model_args = unipelt_arguments.ModelArguments()
training_args = TrainingArguments(output_dir='output/moral_output')

data_train_pd, data_dev_pd, data_test_pd = utils.load_data_complete(train_file=data_args.train_file, dev_file=data_args.validation_file, dev_label_file=data_args.validation_labels_file, test_file=data_args.test_file)
data_train_pd = utils.clean_raw_data(data_train_pd)
data_dev_pd = utils.clean_raw_data(data_dev_pd)
data_test_pd = utils.clean_raw_data(data_test_pd)

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

dataset_emp_train, dataset_dis_train = preprocessing.get_preprocessed_dataset(data_train_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay'])
dataset_emp_dev, dataset_dis_dev = preprocessing.get_preprocessed_dataset(data_dev_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay'])
dataset_emp_test, dataset_dis_test = preprocessing.get_preprocessed_dataset(data_test_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['essay'])

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


def scatter_moral_empdis(pca_features, labels):

    pca_dim = pca_features.shape[1]

    fig, axes = plt.subplots(pca_dim, sharey=True)
    for i in range(pca_dim):
        moral_dim_pc_i = pca_features[:, i]
        axes[i].scatter(labels, moral_dim_pc_i)

    plt.savefig(training_args.output_dir + f'/scatter_moral_{data_args.task_name}.pdf')


#def distance_moral_empdis(pca_features, labels, decimal_count=1):

#    # round label scores, so that the scores are in sort of bins: 
#    labels_bin = np.around(labels, decimal_count)  # sort of discrete space, depending on the decimal count

#    pca_dim = pca_features.shape[1]

#    fig, axes = plt.subplots(pca_dim, sharey=True)
#    for i in range(pca_dim):
#        moral_dim_pc_i = pca_features[:, i]
#        axes[i].scatter(labels, moral_dim_pc_i)

#    plt.savefig(training_args.output_dir + f'/scatter_moral_{data_args.task_name}.pdf')


def plot_moral_empdis(bins, binned_data):
    colors = ['red', 'blue', 'yellow']
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
    plt.xlabel('Average moral score')
    plt.ylabel('Score (in bins)')
    plt.savefig(training_args.output_dir + f'/ave_moral_{data_args.task_name}.pdf')
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
        item_bin_idx = np.intersect1d(min_idx, max_idx)[0]
        moral_pca_i = moral_pca[idx]
        binned_pca[item_bin_idx].append(moral_pca_i)
        binned_labels[item_bin_idx].append(score)
    # remove last bin, because it is 0 anyways, just needed it for the calculation
    binned_pca = binned_pca[:-1]
    binned_labels = binned_labels[:-1]
    bins = bins[:-1]

    return binned_pca, binned_labels, bins


# ------------------
# create moral score
# ------------------
print(train_dataset)
# TODO: Do this with resampled even data
essays = train_dataset['essay']
labels = np.array(train_dataset['label'])

essay_embeddings = sent_model.get_sen_embedding(essays)

mort_pca = load_mort_pca(filename=data_args.data_dir + '/MoRT_projection/projection_model.p')
moral_dim = mort_pca.transform(essay_embeddings)

print(type(moral_dim))
print(moral_dim)

try:
    print(moral_dim.shape)
except:
    pass

pca_dim = moral_dim.shape[1]

for i in range(pca_dim):
    print(f'correlation of PC {i+1}')
    moral_dim_pc_i = moral_dim[:, i]
    print('labels.shape', labels.shape)
    print('moral_dim_pc_i.shape', moral_dim_pc_i.shape)
    r, p = pearsonr(moral_dim_pc_i, labels)
    print(f'r: {r}, p: {p}')
    print()
    
scatter_moral_empdis(moral_dim, labels)

binned_pca, binned_labels, bins = bin_data(labels, moral_dim, 0.1)
plot_moral_empdis(bins, binned_pca)
    
