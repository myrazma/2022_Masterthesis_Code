"""
Script to correlate the ED and DD with subsampled data
Stores the output, for further analysis look at the Notebook emp_dim_analysis.ipynb
"""


from pathlib import Path
import os
import sys
import torch
print(sys.path)
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# get imports from the submodule
sys.path.append(str(path_root))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules/2022_Masterthesis_UnifiedPELT'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../submodules/2022_Masterthesis_UnifiedPELT/transformers')) 
print(sys.path)
import importlib
#import transformers
try:
    unipelt_transformers = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers')
except:
    print('The UniPelt Input is not available. \n The submodule in "submodules.2022_Masterthesis_UnifiedPELT". Not exiting.')
    sys.exit(-1)

try:
    unipelt_utils = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.utils')
    unipelt_preprocessing = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.preprocessing')
    unipelt_arguments = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.arguments')
except:
    print('Utils is the problem.')
    sys.exit(-1)

import importlib
DataTrainingArguments = unipelt_arguments.DataTrainingArguments

from torch import t
import utils.utils as utils
import utils.preprocessing as preprocessing
import utils.feature_creator as feature_creator
from utils.arguments import PCAArguments
from scipy.stats import pearsonr
from EmpDim.pca import DataSelector
import pandas as pd

def correlate_pca(labels, words, pca_args, data_args, device='cpu'):
    fc = feature_creator.FeatureCreator(pca_args=pca_args, data_args=data_args, device=device)
    # create pca features
    pca_features = fc.create_pca_feature(words, task_name=data_args.task_name)
    if pca_features.shape[1] == 1:
        pca_features = pca_features.reshape((-1, 1))

    r, p = pearsonr(labels, pca_features)
    return r, p

def generate_pca_output(output_dir, split, essays, pca_args, data_args, device='cpu'):
    fc = feature_creator.FeatureCreator(pca_args=pca_args, data_args=data_args, device=device)
    # create pca features
    pca_features = fc.create_pca_feature(essays, task_name=data_args.task_name)
    
    df = pd.DataFrame(pca_features)
    output_dir = output_dir + f'/EmpDim/{data_args.task_name}/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = output_dir + f'pca_dim_{split}_{data_args.task_name}.tsv'
    df.to_csv(output_path, index=None, sep='\t')


def run():
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")


    data_args = DataTrainingArguments()
    data_args.task_name = 'distress'

    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=data_args.data_dir)

    if data_args.task_name == 'distress':
        lexicon = distress_lex
    else:
        lexicon = empathy_lex
    data_train_pd, data_dev_pd, data_test_pd = utils.load_data_complete(train_file=data_args.train_file, dev_file=data_args.validation_file, dev_label_file=data_args.validation_labels_file, test_file=data_args.test_file)
    data_train_pd = utils.clean_raw_data(data_train_pd, keep_id=True)
    data_dev_pd = utils.clean_raw_data(data_dev_pd, keep_id=True)
    data_test_pd = utils.clean_raw_data(data_test_pd, keep_id=True)
    datasets = {'train': data_train_pd, 'dev': data_dev_pd, 'test': data_test_pd}
    

    ds = DataSelector()
    subsampled_lex, bins = ds.subsample_even_score_distr(distress_lex, datapoints_per_bin=15, bin_size=0.1, return_bins=True)
    print(f'Amount of subsampled data: {len(subsampled_lex)}')
    values = list([score for word, score in subsampled_lex])

    labels = [item[1] for item in subsampled_lex]
    words = [item[0] for item in subsampled_lex]
    # create the different pca arguments to compare


    pca_args = PCAArguments()
    pca_args.data_lim = 1000
    pca_args.vocab_type = 'mm'
    pca_args.vocab_size=10


    pca_args_1 = pca_args
    pca_args_1.dim = 3
    pca_args_1.use_freq_dist = True
    pca_args_1.freq_thresh=0.000005

    r, p = correlate_pca(labels, words, pca_args_1, data_args, device=device)
    print(f'arg 1. r: {r}, p: {p}')

    pca_args_2 = pca_args
    pca_args_2.dim = 1
    pca_args_2.use_freq_dist = False

    r, p = correlate_pca(labels, words, pca_args_2, data_args, device=device)
    print(f'arg 1. r: {r}, p: {p}')

    pca_args_3 = pca_args
    pca_args_3.dim = 3
    pca_args_3.use_freq_dist = False
    pca_args_3.use_question_template=0.000005

    r, p = correlate_pca(labels, words, pca_args_3, data_args, device=device)
    print(f'arg 1. r: {r}, p: {p}')

    output_dir = data_args.data_dir + '/../output'
    split = ['test']
    if not isinstance(split, list):
        split = [split]

    for s in split:
        if s in dataset.keys():
            dataset = datasets[s]
            generate_pca_output(output_dir=output_dir, split=s, essays=dataset['essay'], pca_args=pca_args, data_args=data_args, device=device)
  

if __name__=='__main__':
    run()