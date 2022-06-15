""" Script for running Unipelt Model with possible feature input
Should capture:
1. Different Input:
    a. Lexicon - word average
    b. Lexicon - PCA
2. Changeable parameters for UniPELT settings (Learning rate, methods, etc.)

In here: use trainer (best from submodule/..UnifiedPELT/transformers), same like in run_emp.py


Can we maybe build a framework for this trainer to use it for other models too? So for the model of / in adapter_BERT
"""

import torch
from scipy.stats import pearsonr, spearmanr

# my modules and scripts
from pathlib import Path
import sys

from torch import t
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils
import preprocessing

class UniPELTMultiinput():

    def __init__(self, feature_dim) -> None:
        
        # This should be UniPELT bert (but without classification head)
        self.bert = None  # TODO

        # Should be the same as the classification head in the transformers library
        self.regressor_head = None  # TODO

        # TODO: Where do I have to add the feature dim?
        pass

    def forward(self, input_ids, attention_masks, features):
        # TODO:
        # Lexical features should be of arbitrary length, can also be None
        # Should I set features to None in head: forward(.., features=None)


        #outputs = self.bert(input_ids, attention_masks)
        #bert_output = outputs[1]

        # concat bert output with multi iput - lexical data
        #after_bert_outputs = self.after_bert(bert_output)

        # combine bert output (after short ffn) with lexical features
        #concat = torch.cat((after_bert_outputs, lexical_features), 1)
        #outputs = self.regressor(concat)
        #return outputs
        pass


def run():

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    #parser = HfArgumentParser((MyArguments, ... what else we need))

    #if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #    # If we pass only one argument to the script and it's the path to a json file,
    #    # let's parse it to get our arguments.
    #    model_args, data_args, training_args, adapter_args = parser.parse_json_file(
    #        json_file=os.path.abspath(sys.argv[1])
    #    )
    #else:
    #    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")

    data_root_folder = 'data/'
    task_name = 'distress'

    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)

    fc = preprocessing.FeatureCreator(data_root_folder=data_root_folder, device=device)

    result = fc.create_pca_feature(data_train_pd['essay'], task_name=task_name)
    print('\n\nPCA features')
    print('\n result: ', result[:10])
    print('\n Distress label: ', data_train_pd[task_name].to_numpy())
    labels = data_train_pd[task_name].to_numpy().reshape(-1)
    emp_dim = result.reshape(-1)
    print('PEARSON R: ', pearsonr(labels, emp_dim))


    print('\n\nLexicon features')
    data_train_pd = preprocessing.tokenize_data(data_train_pd, 'essay')
    data_dev_pd = preprocessing.tokenize_data(data_dev_pd, 'essay')
    
    # create lexical features
    fc = preprocessing.FeatureCreator(data_root_folder=data_root_folder)
    lexicon_rating = fc.create_lexical_feature(data_train_pd['essay_tok'], task_name=task_name)
    print(lexicon_rating)
    print('PEARSON R: ', pearsonr(labels, lexicon_rating))


    print('\n\nPEARSON R lexicon rating and empdim: ', pearsonr(emp_dim, lexicon_rating))
        



if __name__ == '__main__':
    run()