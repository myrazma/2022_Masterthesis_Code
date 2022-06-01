from pickle import FALSE
from pyexpat import model
from random import random
from sqlite3 import Timestamp
from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser
from sentence_transformers import SentenceTransformer

from datasets import Dataset

import torch

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from scipy.optimize import minimize

import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

import tensorboard
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import os
import sys
import pandas as pd
import decimal
import math
import random
from datetime import datetime


# own modules
from funcs_mcm import BERTSentence
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from torch.utils.data import DataLoader
import utils
import preprocessing
import matplotlib.pyplot as plt


START_TS = datetime.timestamp(datetime.now())

@dataclass
class MyArguments:
    data_dir: str = field(
        default='../data/', metadata={"help": "A directory containing the data."}
    )
    task_name: Optional[str] = field(
        default='distress',
        metadata={"help": "The task name to perform the model on. Either distress or empathy."},
    )
    model_name: Optional[str] = field(
        default='',
        metadata={"help": "The transformer model name for loading the tokenizer and pre-trained model."},
    )
    seed: Optional[str] = field(
        default=17,
        metadata={"help": "The seed."},
    )
    vocab_size: Optional[int] = field(
        default=20,
        metadata={"help": "The size of the vocabualry for max min scores."},
    )
    vocab_type: Optional[int] = field(
        default='mmn',
        metadata={"help": "Available types are 'mm' (min max), 'mmn' (min max neutral)."},
    )
    vocab_center_strategy: Optional[int] = field(
        default='soft',
        metadata={"help": "Available types are 'soft', 'hard'."},
    )
    data_lim: Optional[int] = field(
        default=None,
        metadata={"help": "The data limit for the lexicon datadim."},
    )
    dim: Optional[int] = field(
        default=1,
        metadata={"help": "The n_components of the PCA / dimension."},
    )
    store_run: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the run will be stored in json."},
    )
    random_vocab: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the vocabulary will be chosen random."},
    )
    use_tensorboard: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, tensorboard will be used."},
    )
    run_id: str = field(
        default=None,
        metadata={"help": "If True, tensorboard will be used."},
    )
    

class DisDimPCA:

    def __init__(self, n_components, task_name, pca=None, tensorboard_writer=None, model_name='PCA'):
        self.pca = pca  # is None until it is fitted
        self.n_components = n_components
        self.explained_var = None
        self.task_name = task_name
        self.tensorboard_writer = tensorboard_writer
        self.model_name = model_name
        self.vocab_size = None
        self.logging = pd.DataFrame()

    def fit(self, sent_embeddings, transform_embeddings=False):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(sent_embeddings)
        self.explained_var = self.pca.explained_variance_ratio_
        self.vocab_size = len(sent_embeddings)

        if transform_embeddings == True:  # return transformed sentences
            return self.transform(sent_embeddings)

    def fit_transform(self, sent_embeddings):
        return self.fit(sent_embeddings, transform_embeddings=True)

    def transform(self, sent_embeddings):
        return self.pca.transform(sent_embeddings)

    def correlate_dis_dim_scores(self, sent_embeddings, true_scores, printing=True):
        # does pearson r correlation for given sentence embeddings on all
        # possible principal components
        sent_transformed = self.transform(sent_embeddings)
        if isinstance(true_scores, list): true_scores = np.array(true_scores)

        pca_pearsonr, pca_pearsonp = [], []
        for i in range(sent_transformed.shape[1]):
            princ_comp = i
            # version 1:
            sent_transformed_i = sent_transformed[:, i]  # same result as multiplying data with the one principal component
            r, p = pearsonr(sent_transformed_i, true_scores)
            if isinstance(r, list):
                r = r[0]
            pca_pearsonr.append(float(r))
            pca_pearsonp.append(float(p))
            if printing: print(f'\nPC {princ_comp}. r: {r}, p: {p}')

        return pca_pearsonr, pca_pearsonp
    
    def plot_dis_dim_scores(self, sent_embeddings, true_scores, pca_pearsonr=None, title_add_on=''):
        sent_transformed = self.pca.transform(sent_embeddings)
        if isinstance(true_scores, list): true_scores = np.array(true_scores)

        # handling if pearsonr input is not correct
        if pca_pearsonr is not None and len(pca_pearsonr) != len(self.explained_var):
            pca_pearsonr = None
            print('Pearson r does not have the correct size, ignroing this input')

        fig, ax = plt.subplots(len(self.explained_var), figsize=(10,10))
        for i in range(len(self.explained_var)):
            subplot_title = f'PC {i}. Explained var: {self.explained_var[i]:.4f}.'
            if pca_pearsonr is not None: subplot_title = subplot_title + f' Pearson r: {pca_pearsonr[i]:.4f}'
            ax[i].scatter(sent_transformed[:, i], true_scores)
            ax[i].set_box_aspect(1)
            ax[i].set_title(subplot_title)
            ax[i].set_xlabel('PCA dimension projection score')
        plt.ylabel(f'{self.task_name} score')


        filename = f'{self.model_name}_{self.task_name}{title_add_on}'
        plt.savefig(f'EmpDim/plots/{filename}.pdf')
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure(f'Scatter Predictions - {title_add_on}', plt.gcf())
        plt.close()


    def update_log(self,
                    arguments, 
                    pearson_r=None, 
                    pearson_p=None, 
                    explained_var=None, 
                    vocab_min_words=None, 
                    vocab_min_scores=None, 
                    vocab_max_words=None, 
                    vocab_max_scores=None, 
                    vocab_neut_words=None, 
                    vocab_neut_scores=None,
                    ):
        new_row = pd.DataFrame({'id': START_TS,  # can also be trated as the individual id
                'dim': arguments.dim, 
                'task_name': arguments.task_name, 
                'data_lim': arguments.data_lim,
                'vocab_size': arguments.vocab_size,
                'random_vocab': arguments.random_vocab,
                'vocab_min': vocab_min_words,
                'vocab_min_scores': vocab_min_scores,
                'vocab_max': vocab_max_words,
                'vocab_neut_scores': vocab_max_scores,
                'vocab_max': vocab_max_words,
                'vocab_neut_scores': vocab_max_scores,
                'princip_comp': list(range(len(self.explained_var))),  # the number of the principal component
                'pca_var': self.explained_var,  # principal components
                'pca_pearsonr': pearson_r,
                'pca_pearsonp': pearson_p
                })

        df = pd.concat([df, new_row])



class DataSelector:
    # read and load data
    # select data
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.blacklist = ['blanket', 'home', 'shipwreck', 'cub', 'joke', 'fart', 'gag', 'clown'] # or []

    def stem_words(self, words):
        """Stem words, capable of handling different types. words can either be
        of type str
        or type list(str)
        or type list((str, score))
        where the str object will be stemmed

        Args:
            words (list((str, float)) or list(str) or str): The words as described above

        Returns:
            list((str, float)) or list(str) or str: The stemmed words, returned in the same type as the input type
        """
        # words -> list of words
        # or words -> liste of tuples (list(str, float))
        # or words -> str
        if isinstance(words, list):
            if isinstance(words[0], tuple):
                word_stems = [(self.stemmer.stem(w), score) for (w, score) in words]
            elif isinstance(words[0], str):
                word_stems = [self.stemmer.stem(w) for w in words]
            else:
                print('MyWarning in stem_words(): variable "words" should be a list of strings or list of tuples. Returning empty list.')
                word_stems = []
        elif isinstance(words, str):
            word_stems = self.stemmer.stem(words)
        else:
            print('MyWarning in stem_words(): variable "words" should be a list of strings or list of tuples. Returning empty list.')
            word_stems = []
        return word_stems


    def remove_dublicates(self, words, sorting='max'):
        """Remove dublikates from a (sorted) list.

        Args:
            words (list((str, float)) or list(str)): The list of words
            sorting (str, optional): The sorting mechanism. Defaults to 'max'.

        Returns:
            _type_: _description_
        """
        # words -> liste of tuples (list(str, float))
        # the way to sort the scores (item[1])
        distinct_words = []
        if isinstance(words, list):
            if isinstance(words[0], tuple):
                # sort to make sure, we are returning the word with the highest/lowest value
                reverse = True if sorting=='max' else False
                sorted_words = [(word, score) for word, score in sorted(words, key=lambda item: item[1], reverse=reverse)]
                set_words = list(set([word for word, score in sorted_words]))
                for word, score in sorted_words:
                    if word in set_words:
                        set_words.remove(word)
                        distinct_words.append((word, score))
            if isinstance(words[0], str):
                reverse = True if sorting=='max' else False
                set_words = list(set([word for word in words]))
                for word in words:
                    if word in set_words:
                        set_words.remove(word)
                        distinct_words.append((word))

        return distinct_words


    def get_verbs(self, words):
        """Get the words from a list

        Args:
            words (list((str, float)) or list(str)): The list of words

        Returns:
            list((str, float)) or list(str): The verbs
        """
        verbs = []
        if isinstance(words, list):
            if isinstance(words[0], tuple):
                for word, score in words:
                    if word not in self.blacklist:
                        verb_synset_ls = wn.synsets(word, pos=wn.VERB)  # if a verb can be found in the list
                        if len(verb_synset_ls) >= 1:
                            #print(f'{word} is a verb (score: {score})\n Synset: {verb_synset_ls}\n')
                            verbs.append((word, score))
            if isinstance(words[0], str):
                for word in words:
                    if word not in self.blacklist:
                        verb_synset_ls = wn.synsets(word, pos=wn.VERB)
                        if len(verb_synset_ls) >= 1:
                            verbs.append(word)

        return verbs


    def select_words(self, lexicon, word_count, random_vocab=False, samples=['min', 'max'], center_strategy='soft'):
        # samples can be ['min', 'max', 'neut']
        # or ['min', 'max']
        # return min and max sorted words with length of word_count
        if samples is None:  # stardard: return min and max score
            samples = ['min', 'max']
        print(f'Vocabulary info: \n word count: {word_count}')
        def __get_words(words, sorting):
            reverse = True if sorting=='max' else False
            words_sorted = [(word, score) for word, score in sorted(words, key=lambda item: item[1], reverse=reverse)]
            word_stems = self.stem_words(words_sorted)
            distinct_word_stems = self.remove_dublicates(word_stems, sorting=sorting)
            distinct_verbs = self.get_verbs(distinct_word_stems)
            distinct_verbs_selected = distinct_verbs
            # shuffle for random
            if random_vocab:
                distinct_verbs_selected = distinct_verbs.copy()
                random.shuffle(distinct_verbs_selected)
            print(f'Mode: {sorting}')
            print(f'Range from {min([score for word, score in distinct_verbs_selected])} to {max([score for word, score in distinct_verbs_selected])}')
            return distinct_verbs_selected
        words = [(key, lexicon[key]) for key in lexicon.keys()]
        words_min = __get_words(words, sorting='min')[:word_count]
        words_max = __get_words(words, sorting='max')[:word_count]

        # if neutral word requested, also get the neutral words around the center
        if 'neut' in samples:

            # - get the center -
            center = 0
            possible_strategies = ['soft', 'hard']
            if center_strategy not in possible_strategies:
                print(f'\n MyWarning (select_words): The strategy to find the center {center_strategy} is unknown. Known: {possible_strategies}. strategy: soft will be used.\n')
                center_strategy = 'soft'

            if center_strategy == 'soft':  
                # taking the highest score of the min values and the lowest score of the high values to find the middle
                lower_bound = max([item[1] for item in words_min])
                upper_bound = min([item[1] for item in words_max])
                center = (lower_bound + upper_bound) / 2
            elif center_strategy == 'hard':
                # taking the lowest score of possible values and the highest score of possible values to find the middle
                lower_bound = min([item[1] for item in words])
                upper_bound = max([item[1] for item in words])
                center = (lower_bound + upper_bound) / 2
            print('Neutr Center:', center)

            # - divide words into two lists: smaller than the center, bigger than the center -
            smaller_center = [item for item in words if item[1] <= center]
            bigger_center = [item for item in words if item[1] > center]

            half_word_count = int(word_count/2)
            # for smaller than center use max sorting (the ones close to the center)
            words_neutr_smaller = __get_words(smaller_center, sorting='max')[:half_word_count]
            # for bigger than center use min sorting (the ones close to the center)
            words_neutr_bigger = __get_words(bigger_center, sorting='min')[:half_word_count]

            # - combine vocabulary to get the neutral vocabulary -
            words_neutr = words_neutr_smaller + words_neutr_bigger
            
            return words_min, words_max, words_neutr
        else:
            return words_min, words_max


    def subsample_even_score_distr(self, lexicon_input, datapoints_per_bin, bin_size, return_binned_data=False, return_bins=False):
        """Subsample data from a lexicon

        Args:
            lexicon (_type_): _description_
            datapoints_per_bin (_type_): _description_
            bin_size (_type_): _description_
            return_binned_data (bool, optional): If set to True, the data will be returned as a 
                                        two dimensional list, stored in ther bins. If false, a 
                                        one dimensional list will be returned. Defaults to False.
        """
        # - create list of tuples: list((str, float)) -
        if isinstance(lexicon_input, dict):
            words_sorted = [(word, score) for word, score in sorted(lexicon_input.items(), key=lambda item: item[1])]
        elif isinstance(lexicon_input, list):  # if list of tuples
            if isinstance(lexicon_input[0], tuple):
                words_sorted = [(word, score) for word, score in sorted(lexicon_input, key=lambda item: item[1])]
        else:  # else, wecan't handle that right now
            print('MyError: The input should be a lexicon or list of tuples -> list((str, float)). Not implement for other types.')
            sys.exit(-1)

        # - create bins -
        decimal_count = abs(decimal.Decimal(str(bin_size)).as_tuple().exponent)
        min_score = min([item[1] for item in words_sorted])
        max_score = max([item[1] for item in words_sorted])
        bins_start = math.floor(min_score * (10**decimal_count)) / (10**decimal_count)
        bins_end = math.ceil(max_score * (10**decimal_count)) / (10**decimal_count)
        # add the end point to the bins as well, to get the upper range for the elements
        # this will be removed later on, since it is not actually a bin
        bins = np.arange(bins_start, bins_end + bin_size, bin_size)

        # - divide data into bins - 
        binned_data = [[] for i in range(len(bins))]
        for word, score in words_sorted:
            min_idx = np.where(bins <= score)[0]
            max_idx = np.where(bins > score)[0] - 1
            item_bin_idx = np.intersect1d(min_idx, max_idx)[0]
            binned_data[item_bin_idx].append((word, score))
        # remove last bin, because it is 0 anyways, just needed it for the calculation
        binned_data = binned_data[:-1]
        bins = bins[:-1]

        # - shuffle the bins -
        for bin in binned_data:
            random.shuffle(bin)
            
        # - select data points from those bins of size <datapoints_per_bin> -
        binned_data = [bin[:datapoints_per_bin] for bin in binned_data]
        
        if return_binned_data:
            return binned_data if not return_bins else (binned_data, bins)

        unbinned_data = [item for bin in binned_data for item in bin]
        return unbinned_data if not return_bins else (unbinned_data, bins)


def save_run_deprecated(my_args, vocab_min, vocab_max, pca_var, pca_pearsonr, pca_pearsonp, filename=None, tensorboard_writer=None):
    if filename == None:
        filename = 'empdim_settings.csv'

    df = pd.DataFrame()

    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)

    pca_var = list([float(val) for val in pca_var.reshape(-1)])
    pca_pearsonr = list(pca_pearsonr)
    pca_pearsonp = list(pca_pearsonp)
    vocab_min_words = ';'.join([item[0] for item in vocab_min])
    vocab_min_scores = ';'.join([str(item[1]) for item in vocab_min])
    vocab_max_words = ';'.join([item[0] for item in vocab_max])
    vocab_max_scores = ';'.join([str(item[1]) for item in vocab_max])

    ts = datetime.timestamp(datetime.now())

    #for dim, (var, r, p) in enumerate(zip(pca_var, pca_pearsonr, pca_pearsonp)):
    new_row = pd.DataFrame({'id': START_TS,  # can also be trated as the individual id
                'dim': my_args.dim, 
                'task_name': my_args.task_name, 
                'data_lim': my_args.data_lim,
                'vocab_size': my_args.vocab_size,
                'random_vocab': my_args.random_vocab,
                'vocab_min': vocab_min_words,
                'vocab_min_scores': vocab_min_scores,
                'vocab_max': vocab_max_words,
                'vocab_max_scores': vocab_max_scores,
                'princip_comp': list(range(len(pca_pearsonr))),  # the number of the principal component
                'pca_var': pca_var,  # principal components
                'pca_pearsonr': pca_pearsonr,
                'pca_pearsonp': pca_pearsonp
                })

    #if tensorboard_writer is not None:
    #    with SummaryWriter() as w:
    #        for i in range(len(pca_pearsonr)):
    #            tensorboard_writer.add_hparams(new_row_dict, {'princip_comp': int(i), 'pca_var': float(pca_var[i]), 'pca_pearsonr': float(pca_pearsonr[i]), 'pca_pearsonp': pca_pearsonp[i]})
    df = pd.concat([df, new_row])
    print('Saved Information:\n', new_row[:5])
    
    df.to_csv(filename, sep=',')


def save_run(data_dict, filename=None, tensorborad_writer=None):
    if filename == None:
        filename = 'empdim_settings.csv'

    df = pd.DataFrame()

    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)

    new_row = pd.DataFrame(data_dict)
    df = pd.concat([df, new_row])
    print('Saved Information:\n', new_row[:5])
    
    df.to_csv(filename, sep=',')





def run():
    
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")

    # get arguments
    parser = HfArgumentParser(MyArguments)
    my_args = parser.parse_args_into_dataclasses()[0]

    if my_args.task_name not in ['empathy', 'distress']:
        print("task name not available, choose 'empathy' or 'distress' next time. Usign empathy now")
        my_args.task_name = 'empathy'

    torch.manual_seed(my_args.seed)
    random.seed(my_args.seed)

    run_history = pd.DataFrame()

    tensorboard_writer = SummaryWriter('/emp_dim_runs') if my_args.use_tensorboard else None

    # ------------------------
    #     Load the lexicon 
    # ------------------------
    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=my_args.data_dir)

    if my_args.task_name == 'distress':
        lexicon = distress_lex
    else:
        lexicon = empathy_lex
    print(f'Task name: {my_args.task_name}')

    sorted_lexicon = words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
    score_range = (1, 7)
    actual_score_range = (min([item[1] for item in sorted_lexicon]), max([item[1] for item in sorted_lexicon]) )
    
    # setup data selector class
    data_selector = DataSelector()
    
    # --- normalize the scores in the lexicon ---
    # lexicon = dict([(key, lexicon[key] / score_range[1]) for key in lexicon])

    # --- select the most representative words for low and high distress and empathy ---
    # n words with highest and words with lowest ranking values
    def scatter_vocab(vocab, title):
        scores = [item[1] for item in vocab]
        random_y = [random.uniform(0, 1) for i in range(len(scores))]
        plt.scatter(scores, random_y, label=title)
        plt.ylabel('random values')
        plt.xlabel('empathy / distress score')
        plt.legend(loc=3)
        plt.savefig(f'EmpDim/plots/PCA_{title}.pdf')
        plt.close()

    #score_min, score_max = data_selector.select_words(lexicon, my_args.vocab_size, random_vocab=my_args.random_vocab, samples=['min', 'max'])
    #print('min | max')
    #print('score min \n', score_min)
    #print('score max \n', score_max)
    #vocab = score_min + score_max
    #scatter_vocab(vocab, 'min_max')

    # --- min max neutr, soft center ---
    #score_min, score_max, score_neutr = data_selector.select_words(lexicon, my_args.vocab_size, random_vocab=my_args.random_vocab, samples=['min', 'max', 'neut'])
    #print('min | max | neutr - soft')
    #print('score min \n', score_min)
    #print('score max \n', score_max)
    #print('score neutr \n', score_neutr)
    #vocab = score_min + score_max + score_neutr
    #scatter_vocab(vocab, 'min_max_neutr_soft')

    # --- min max neutr, hard center ---
    score_min, score_max, score_neutr = data_selector.select_words(lexicon, my_args.vocab_size, random_vocab=my_args.random_vocab, samples=['min', 'max', 'neut'], center_strategy=my_args.vocab_center_strategy)
    print('min | max | neutr - hard')
    print('score min \n', score_min)
    print('score max \n', score_max)
    print('score neutr \n', score_neutr)
    vocab = score_min + score_max + score_neutr
    random.shuffle(vocab)
    scatter_vocab(vocab, 'min_max_neutr_hard')

    vocab_sentences = [item[0] for item in vocab] # get sentences
    vocab_labels = np.array([item[1] for item in vocab]).reshape(-1, 1) # get the labels

    # --- init BERTSentence for sentence embeddings ---
    # Use Sentence embeddings like MoRT (using their code and functions in funcs_mcm.py): https://github.com/ml-research/MoRT_NMI/blob/master/MoRT/mort/funcs_mcm.py
    # Schramowski et al., 2021, Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do
    sent_model = BERTSentence(device=device) #, transormer_model='paraphrase-MiniLM-L6-v2') # TODO use initial model (remove transformer model varibale from head)
    
    # get the sentence embeddings of the vocabulary
    vocab_embeddings = sent_model.get_sen_embedding(vocab_sentences)
    
    # ------------------------------------
    #   Do PCA with the vocab embeddings
    # ------------------------------------
    print('------------------ Start PCA ------------------')
    dim_pca = DisDimPCA(n_components=my_args.dim, task_name=my_args.task_name)
    transformed_emb = dim_pca.fit_transform(vocab_embeddings)

    princ_comp_idx = 0  # principal component: 0 means first

    # get eigenvectors from best / highest eigenvalue
    eigen_vec = dim_pca.pca.components_
    projection = eigen_vec[princ_comp_idx]  # TODO check if this line is really correct, am I selecting and getting the right values?
    print(projection.shape)
    # ------------------------------
    #    Analyse the PCA outcome
    # ------------------------------

    # --- Get principal component / Eigenvector ---
    # - How much variance (std..) do they cover? -
    # - how many are there?
    var = dim_pca.explained_var
    print(list(var))
    pca_dim = transformed_emb[:, 0]

    plt.scatter(pca_dim, vocab_labels)
    plt.ylabel(f'{my_args.task_name} score')
    plt.xlabel('PCA dim')
    plt.title(f'PC1 covering {var[princ_comp_idx]}')
    plt.savefig('EmpDim/plots/PCA_dim.pdf')
    plt.close()
    
    # plot cumsum
    plt.plot(var.cumsum())
    plt.xticks(list(range(len(var))))
    plt.savefig('EmpDim/plots/PCA_var_cumsum.pdf')    
    plt.close()


    # --- check projection on all words datatset (all words from lexicon) ---

    # --- transform lexicon into list of tuples ---
    all_words_n_scores = [(key, lexicon[key]) for key in lexicon]

    # --- get whole lexicon and limit by data limit ---
    print('correlate with original shuffled data')
    all_words_n_scores_rand = all_words_n_scores
    random.shuffle(all_words_n_scores_rand)  # shuffle words
    all_words_n_scores_rand = all_words_n_scores_rand[:my_args.data_lim]
    print('Dataset size:', len(all_words_n_scores_rand))
    all_words_rand = [item[0] for item in all_words_n_scores_rand]
    all_words_rand_labels = np.array([item[1] for item in all_words_n_scores_rand]).reshape(-1, 1)
    # get their sentence embeddings
    # get correlation and plot
    all_words_rand_embeddings = sent_model.get_sen_embedding(all_words_rand)
    r_rand, p_rand = dim_pca.correlate_dis_dim_scores(all_words_rand_embeddings, all_words_rand_labels, printing=True)
    dim_pca.plot_dis_dim_scores(all_words_rand_embeddings, all_words_rand_labels, r_rand, title_add_on='all_words_rand')

    # --- correlate for evenly sampled data samples ---
    datapoints_per_bin = 15
    print(f'correlate for even subsamples {datapoints_per_bin}')
    even_subsamples = data_selector.subsample_even_score_distr(all_words_n_scores, datapoints_per_bin=datapoints_per_bin, bin_size=0.1)
    sentences_input = [item[0] for item in even_subsamples]
    embedding_input = sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    true_labels = [item[1] for item in even_subsamples]
    r_even_15, p_rand_even15 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_even_15, title_add_on='all_words_even')
    
    # --- correlate for evenly sampled data samples ---
    datapoints_per_bin = 10
    print(f'correlate for even subsamples {datapoints_per_bin}')
    even_subsamples = data_selector.subsample_even_score_distr(all_words_n_scores, datapoints_per_bin=datapoints_per_bin, bin_size=0.1)
    sentences_input = [item[0] for item in even_subsamples]
    embedding_input = sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    true_labels = [item[1] for item in even_subsamples]
    r_even_15, p_rand_even15 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_even_15, title_add_on='all_words_even')


    lim = 100
    print(f'\n Do correlation on the words with {lim} min an max scores (insg. {lim*2} words). Add random words from middle: {lim}')
    # get min 100 and max 100 of the words
    # with random
    words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
    words_random = words_sorted[lim:-lim]
    random.shuffle(words_random)
    words_random = words_random[:lim]

    words_min = words_sorted[:lim]
    words_max = words_sorted[-lim:]
    sentences_input = [item[0] for item in words_min + words_max + words_random]
    embedding_input = sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    true_labels = [item[1] for item in words_min + words_max + words_random]
    r_twotailed_random_100, p_rand_even15 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_twotailed_random_100, title_add_on='all_words_twotailed_random_100')
    # without random

    print(f'\n Do correlation on the words with {lim} min an max scores (insg. {lim*2} words)')
    words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
    words_min = words_sorted[:lim]
    words_max = words_sorted[-lim:]
    sentences_input = [item[0] for item in words_min + words_max]
    embedding_input = sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    true_labels = [item[1] for item in words_min + words_max]
    r_twotailed_100, p_rand_even15 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_twotailed_100, title_add_on='all_words_twotailed_100')


    save_run(run_history)
    # ------------------------------
    #    Apply PCA to the essays
    # ------------------------------
    """
    # TODO
    print('\n Apply PCA to the essays \n')
    # --- preprocess data ---
    # - load data -
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=my_args.data_dir)
    data_train_pd = utils.clean_raw_data(data_train_pd[:200])
    data_dev_pd = utils.clean_raw_data(data_dev_pd[:200])

    train_sent = sent_model.get_sen_embedding(data_train_pd['essay'])
    train_sent_transformed = pca.transform(train_sent)  
    train_labels = np.array(data_train_pd[my_args.task_name]).reshape(-1, 1)

    for i in range(train_sent_transformed.shape[1]):
        #print(all_word_dataset_transformed.shape[1])
        princ_comp = i
        print(f'principal component {princ_comp}')
        # version 1:
        train_sent_transformed_i = train_sent_transformed[:, i]  # same result as multiplying data with the one principal component
        r, p = pearsonr(train_sent_transformed_i, train_labels)
        print('r', r)
        print('p', p)
    """

                                  

    # essay encoded
    #print(dataset_train)
    #train_input_ids = torch.tensor(np.array(dataset_train["input_ids"]).astype(int))
    #train_attention_mask = torch.tensor(np.array(dataset_train["attention_mask"]).astype(int))
    #train_labels = torch.tensor(np.array(dataset_train["label"]).astype(np.float32).reshape(-1, 1))

    # ---
    # use other sentence mbeddings





    # - encode data -
    # - tranform data -
    # --- analyse data ---
    # - correlate this score with the actual label -




if __name__ == '__main__':
    run()