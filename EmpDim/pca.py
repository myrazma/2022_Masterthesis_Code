"""Script for creating and evaluating pca
Running this script (run()) will create a pca with the vocabulary based on the input parameters

This script can also be used from outside by using the classes
DisDimPCA:
    The pca class
create_pca():
    Load pca model, if available.
    Otherwise: Creates and fits a pca model based on the input.
evaluate_pca():
    Evaluating pca with lexical data.
"""

from pickle import FALSE
from pyexpat import model
from random import random
from sqlite3 import Timestamp
from regex import R
from sklearn import metrics
from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser
from sentence_transformers import SentenceTransformer
import pickle
import pathlib

from datasets import Dataset

import torch

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.corpus import brown, reuters, gutenberg
nltk.download('gutenberg')
nltk.download('reuters')
nltk.download('brown')
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
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


# own modules
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from EmpDim.funcs_mcm import BERTSentence
from torch.utils.data import DataLoader
import utils.utils as utils
import utils.preprocessing as preprocessing
from utils.arguments import PCAArguments, DataTrainingArguments

this_file_path = str(pathlib.Path(__file__).parent.resolve())

ID = datetime.now().strftime("%Y-%m-%d_%H%M%S")
print(ID)

    

class DisDimPCA:
    """Class for the distress / empathy dimension pca
    Has all attributes needed to create the pca including the sentence bert model (Need to use the same model, otherwise the transformation isnt working anyways)
    """

    def __init__(self, n_components, task_name, pca=None, tensorboard_writer=None, model_name='PCA', device='cpu'):
        self.pca = pca  # is None until it is fitted
        self.n_components = n_components
        self.explained_var = None
        self.task_name = task_name
        self.tensorboard_writer = tensorboard_writer
        self.model_name = model_name
        self.vocab_size = None
        self.logging = pd.DataFrame()

        # --- init BERTSentence for sentence embeddings ---
        # Use Sentence embeddings like MoRT (using their code and functions in funcs_mcm.py): https://github.com/ml-research/MoRT_NMI/blob/master/MoRT/mort/funcs_mcm.py
        # Schramowski et al., 2021, Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do
        self.sent_model = BERTSentence(device=device) #, transormer_model='paraphrase-MiniLM-L6-v2') # TODO use initial model (remove transformer model varibale from head)

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

            sr, sp = spearmanr(sent_transformed_i, true_scores)
            if printing: print(f'\nPC spearmanr {princ_comp}. sr: {sr}, sp: {sp}')

        return pca_pearsonr, pca_pearsonp
    
    def plot_dis_dim_scores(self, sent_embeddings, true_scores, pca_pearsonr=None, title_add_on='', plot_dir='plots/'):
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
            #ax[i].set_box_aspect(1)
            ax[i].set_title(subplot_title)
            ax[i].set_ylabel(f'{self.task_name} score')
        plt.xlabel('PCA dimension projection score')


        filename = f'{self.model_name}_{self.task_name}{title_add_on}'
        plt.savefig(f'EmpDim/{plot_dir}{filename}.pdf')
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure(f'Scatter Predictions - {title_add_on}', plt.gcf())
        plt.close()

    def scatter_vocab_words(self, vocab, x=None, colormap='pink', title_add_on='', ylabel='', xlabel='', set_y_random=True, use_cmap=True, plot_dir='plots/'):
        
        def normalize(vals, scale=None):
            if scale is None:
                scale = (min(vals), max(vals))
            return (vals - (scale[0])) / (scale[1] - (scale[0]))

        calc_dist = lambda a, b: abs(a - b)

        offset = 0.01
        scores = [item[1] for item in vocab]
        scores = np.array(scores)
        words = [item[0] for item in vocab]
        none_random = True  # variable if x or y are non random. If one of them is random, set to False


        y = scores
        y_rand = np.array([random.uniform(0, 1) for y in range(len(y))])
        if ylabel == '':
            ylabel = 'Human Scores'
        if set_y_random:
            none_random = False
            ylabel = ''
            y = y_rand

        if x is None:
            if set_y_random:
                print('MyWarning (scatter_vocab_words): x and y are set to random. One of them should not be random!!')
            x = np.array([random.uniform(0, 1) for y in range(len(vocab))])
        else:
            xlabel = f'{self.task_name} dimension'

        if use_cmap:
            cm = plt.cm.get_cmap(colormap)
            y_norm = normalize(scores)
            x_norm = normalize(x)
            dist = calc_dist(y_norm, x_norm)
            sc = plt.scatter(x, y, c=dist, vmin=0, vmax=max(dist), s=35, cmap=cm)
            # colorbar means übereinstimmung der scores 
            plt.colorbar(sc)       
        else:
            sc = plt.scatter(x, y)
            

        for i, word in enumerate(words):
            plt.annotate(word, (x[i]+offset, y[i]+offset))

        #plt.ylim(0-offset,1+offset)
        #plt.xlim(0-offset,1+offset)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(f'{self.task_name} Dimension - Vocabulary for PCA \n the color represents the smilarity to the human scores from the lexicon (0 means simsilar, 1 not similar)')
        filename = f'PCA_Vocab_{self.model_name}{title_add_on}'
        plt.savefig(f'EmpDim/{plot_dir}{filename}.pdf')

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure(f'Scatter Words - {title_add_on}', plt.gcf())
        plt.close()


    def update_log(self,
                    arguments,
                    task_name, 
                    pearson_r=None, 
                    pearson_p=None, 
                    vocab=None,
                    center=None,
                    note=None,
                    data_len=None
                    ):
        if self.explained_var is None:
            print('MyWarning (update_log): PCA is not fittet yet, data will not be logged')
            return

        if vocab is not None:
            vocab_sorted = [(word, score) for word, score in sorted(vocab, key=lambda item: item[1])]
            vocab_words = ';'.join([item[0] for item in vocab_sorted])
            vocab_scores = ';'.join([str(item[1]) for item in vocab_sorted])   
                
        new_row = pd.DataFrame({'id': str(ID),  # can also be trated as the individual id
                'dim': arguments.dim,
                'task_name': task_name,
                'data_len': data_len,
                'vocab_size': arguments.vocab_size,
                'random_vocab': arguments.random_vocab,
                'vocab_center_strategy': arguments.vocab_center_strategy,
                'center': center,
                'vocab_bin_size': arguments.vocab_bin_size,
                'note':note,
                'use_freq_dist': arguments.use_freq_dist,
                'freq_thresh': arguments.freq_thresh,
                'vocab_type': arguments.vocab_type,
                'vocab_words': vocab_words,
                'vocab_scores': vocab_scores,
                'princip_comp': list(range(len(self.explained_var))),  # the number of the principal component
                'pca_var': self.explained_var,  # principal components
                'pca_pearsonr': pearson_r,
                'pca_pearsonp': pearson_p,
                })

        self.logging = pd.concat([self.logging, new_row])
        if self.tensorboard_writer is not None:
            hparam_dict = {'id': ID,  # can also be trated as the individual id
                    'dim': arguments.dim, 
                    'task_name': task_name, 
                    'data_len': data_len,
                    'vocab_size': arguments.vocab_size,
                    'random_vocab': arguments.random_vocab,
                    'vocab_center_strategy': arguments.vocab_center_strategy,
                    'center': center,
                    'vocab_bin_size': arguments.vocab_bin_size,
                    'note':note,
                    'use_freq_dist': arguments.use_freq_dist,
                    'freq_thresh': arguments.freq_thresh,
                    'vocab_type': arguments.vocab_type,
                    'vocab_words': vocab_words,
                    'vocab_scores': vocab_scores}
            with self.tensorboard_writer as w:
                for i in range(len(self.explained_var)):
                    metrics_dict = {'princip_comp': np.array(range(len(self.explained_var)))[i], 
                                    'pca_var': np.array(self.explained_var)[i], 
                                    'pca_pearsonr': np.array(pearson_r)[i], 
                                    'pca_pearsonp': np.array(pearson_p)[i]}
                    self.tensorboard_writer.add_hparams(hparam_dict, metrics_dict)
        


class DataSelector:
    # read and load data
    # select data
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.blacklist = ['blanket', 'home', 'shipwreck', 'cub', 'joke', 'fart', 'gag', 'clown'] # or []
        self.__word_fdist = None
        self.vocab_center = None

        self.question_template = []

    def get_fdist(self):
        """Create Frequency dictionary for 

        Returns:
            _type_: _description_
        """
        if self.__word_fdist is None:
            corpus = reuters.words() + brown.words() + gutenberg.words()
            self.__word_fdist = FreqDist(word.lower() for word in corpus if word.isalnum())
        return self.__word_fdist

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


    def remove_dublicates(self, words, sorting='max', strategy='sorting'):
        """Remove dublikates from a (sorted) list.

        Args:
            words (list((str, float)) or list(str)): The list of words
            sorting (str, optional): The sorting mechanism. Defaults to 'max'.
            startegy (str, optional): The strategy to remove dublicates. Can be 'mean', or 'sorting'.
                                    Mean will take the mean of words, sorting will take the max / min score, based on what is sorting. Defaults to 'mean'.

        Returns:
            _type_: _description_
        """
        # words -> liste of tuples (list(str, float))
        # the way to sort the scores (item[1])
        try:
            print('Before removing dublicate:', len(words))
        except:
            print()
        distinct_words = []
        if strategy == 'sorting':
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
        if strategy == 'mean':

            word_d = {}
            if isinstance(words, list):
                if isinstance(words[0], tuple):
                    # sort to make sure, we are returning the word with the highest/lowest value
                    reverse = True if sorting=='max' else False
                    sorted_words = [(word, score) for word, score in sorted(words, key=lambda item: item[1], reverse=reverse)]
                    set_words = list(set([word for word, score in sorted_words]))
                    for word, score in sorted_words:
                        if word not in word_d.keys():
                            word_d[word] = [score]
                        else:
                            word_d[word].append(score)
                    for key, value in word_d.items():
                        distinct_words.append((key, np.mean(np.array(value))))

                if isinstance(words[0], str):
                    print('not implemented')


        try:
            print('After removing dublicate:', len(distinct_words))
        except:
            print()
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

        
        try:
            print('After selecting only verbs:', len(verbs))
        except:
            print()

        return verbs


    def select_words(self, lexicon, word_count, random_vocab=False, samples=['min', 'max'], center_strategy='soft', use_freq_dist=False, freq_thresh=0.000002):
        """Selects the vocabulary and returns a list for each selection
        len(output) == len(samples), so the sorting strategies

        Args:
            lexicon (_type_): _description_
            word_count (_type_): _description_
            random_vocab (bool, optional): _description_. Defaults to False.
            samples (list, optional): _description_. Defaults to ['min', 'max'].
            center_strategy (str, optional): _description_. Defaults to 'soft'.
            use_freq_dist (bool, optional): _description_. Defaults to False.
            freq_thresh (float, optional): _description_. Defaults to 0.000002.

        Returns:
            _type_: list with varying size of sub vocabularies, depending on the samples / sorting input len(samples) == len(result)
        """
        # samples can be ['min', 'max', 'neut']
        # or ['min', 'max']
        # return min and max sorted words with length of word_count
        if samples is None:  # stardard: return min and max score
            samples = ['min', 'max']
        result = []  # the result lists

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

            if use_freq_dist:
                # select from word fequency
                fdist = self.get_fdist()
                distinct_verbs_selected = [(word, score) for word, score in distinct_verbs if fdist.freq(word.lower()) >= freq_thresh]

            return distinct_verbs_selected

        words = [(key, lexicon[key]) for key in lexicon.keys()]
        for sorting in samples:
            if sorting == 'min' or sorting == 'max':  # if sorting is neut, than we will handle the case somewhere else
                words_selected = __get_words(words, sorting=sorting)[:word_count]
                result.append(words_selected)
       
        # ------ neutral ------
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
                if 'min' in samples:
                    words_min = result[samples.index('min')]  # get the result array of the list with the minimum verbs
                    lower_bound = max([item[1] for item in words_min])
                else:  # if no min used, than use hard boundary
                    print('MyWarning: No min sorting used. Use hard boundary instead of soft')
                    lower_bound = min([item[1] for item in words])

                if 'max' in samples:
                    words_max = result[samples.index('max')]  # get the result array of the list with the minimum verbs
                    upper_bound = min([item[1] for item in words_max])
                else:  # if no max used, than use hard boundary
                    print('MyWarning: No max sorting used. Use hard boundary instead of soft')
                    upper_bound = max([item[1] for item in words])
                center = (lower_bound + upper_bound) / 2

            elif center_strategy == 'hard':
                # taking the lowest score of possible values and the highest score of possible values to find the middle
                lower_bound = min([item[1] for item in words])
                upper_bound = max([item[1] for item in words])
                center = (lower_bound + upper_bound) / 2
            print('Neutr Center:', center)
            self.vocab_center = center

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
            result.append(words_neutr)

        # list with varying size of sub vocabularies, depending on the samples / sorting input len(samples) == len(result)
        return result  


    def subsample_even_score_distr(self, lexicon_input, datapoints_per_bin, bin_size, return_binned_data=False, return_bins=False):
        """Subsample data from a lexicon

        Args:
            lexicon (list((str, float))) or dict(str: float): The lexicon input either as dictionary -> word: score or list -> (word, score)
            datapoints_per_bin (int): The data points to select per bin. f.e. 10 would return 10 per bins
            bin_size (float): The bin size, f.e. 0.1 is a good size for this data
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
            else:
                print('MyError: The input is a list, but the items are not tulpes. Should be dict or list of tuples -> list((str, float)). Not implement for other types.')
                sys.exit(-1)
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


def save_run(data_dict, filename=None, tensorboard_writer=None):
    if filename == None:
        filename = 'empdim_results.csv'

    df = pd.DataFrame()

    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0)

    new_row = pd.DataFrame(data_dict)
    df = pd.concat([df, new_row])
    print('Saved Information:\n', new_row[:5])
    
    df.to_csv(filename, sep=',')


def scatter_vocab(vocab, title, plot_dir='plots/'):
        scores = [item[1] for item in vocab]
        random_y = [random.uniform(0, 1) for i in range(len(scores))]
        plt.scatter(scores, random_y, label=title)
        plt.ylabel('random values')
        plt.xlabel('empathy / distress score')
        plt.legend(loc=3)
        plt.savefig(f'EmpDim/{plot_dir}PCA_{title}.pdf')
        plt.close()



def create_pca(my_args, data_args, tensorboard_writer=None, return_vocab=False, data_selector=None, device='cpu'):
    # ------------------------------------
    # ------------------------------------
    #        Load pca if available 
    # ------------------------------------
    # ------------------------------------
    pca_exists = False
    pca_dir = this_file_path + f'/../EmpDim/{data_args.task_name}/'
    pca_file_path = pca_dir + 'emp_dim_pca_projection.p'
    if not os.path.exists(pca_dir):
        os.mkdir(pca_dir)
        # load pca
    try:
        dim_pca = pickle.load(open(pca_file_path,'rb'))
        pca_exists = True
        if dim_pca.n_components_ < my_args.dim:  # if the dimension is not high enough, we need to create a new pca with higher n_components
            pca_exists = False
    except:
        print(f'Could not load the pca at file {pca_file_path}.')
        pca_exists = False
    

    # ------------------------------------
    # ------------------------------------
    #           Create Vocab
    # ------------------------------------
    # ------------------------------------
    # ------------------------
    #     Load the lexicon 
    # ------------------------
    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=data_args.data_dir)

    if data_args.task_name == 'distress':
        lexicon = distress_lex
    else:
        lexicon = empathy_lex

    print(f'Task name: {data_args.task_name}')
    sorted_lexicon = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
    score_range = (1, 7)
    actual_score_range = (min([item[1] for item in sorted_lexicon]), max([item[1] for item in sorted_lexicon]) )
    
    # setup data selector class
    if data_selector is None:
        data_selector = DataSelector()
    
    # --- normalize the scores in the lexicon ---
    # lexicon = dict([(key, lexicon[key] / score_range[1]) for key in lexicon])

    # --- select the most representative words for low and high distress and empathy ---
    # n words with highest and words with lowest ranking values
    
    # -----------------------------
    #     Select the vocabulary 
    # -----------------------------
    print('\n ------------------ Vocabulary info ------------------')
    if my_args.vocab_type == 'mmn':  # min, max, neutral
        sample_type = ['min', 'max', 'neut']  # max, min, neutral (with words from center) -> ! decide for center strategy !
        score_min, score_max, score_neutr = data_selector.select_words(lexicon, my_args.vocab_size, random_vocab=my_args.random_vocab, samples=sample_type, center_strategy=my_args.vocab_center_strategy, use_freq_dist=my_args.use_freq_dist, freq_thresh=my_args.freq_thresh)
        print('Mode: min | max | neut ')
        print(f'score min count: {len(score_min)}  \n Range from {min([score for word, score in score_min])} to {max([score for word, score in score_min])} \n{score_min}')
        print(f'score max count: {len(score_max)}  \n Range from {min([score for word, score in score_max])} to {max([score for word, score in score_max])} \n{score_max}')
        print(f'score neutr count: {len(score_neutr)}  \n Range from {min([score for word, score in score_neutr])} to {max([score for word, score in score_neutr])} \n{score_neutr}')
        vocab = score_min + score_max + score_neutr
    elif my_args.vocab_type == 'mm':  # max, min
        sample_type = ['min', 'max']
        score_min, score_max = data_selector.select_words(lexicon, my_args.vocab_size, random_vocab=my_args.random_vocab, samples=sample_type, use_freq_dist=my_args.use_freq_dist, freq_thresh=my_args.freq_thresh)
        print('Mode: min | max')
        print(f'score min count: {len(score_min)}  \n Range from {min([score for word, score in score_min])} to {max([score for word, score in score_min])} \n{score_min}')
        print(f'score max count: {len(score_max)}  \n Range from {min([score for word, score in score_max])} to {max([score for word, score in score_max])} \n{score_max}')
        vocab = score_min + score_max
    elif my_args.vocab_type == 'range':  # words from the whole range
        datapoints_per_bin = my_args.vocab_size  # for type range, the vocabulary size is the amount of data per bin
        bin_size = my_args.vocab_bin_size
        sample_type = ['min']
        print('Mode: range')
        verbs_sorted = data_selector.select_words(lexicon, None, random_vocab=my_args.random_vocab, samples=sample_type, use_freq_dist=my_args.use_freq_dist, freq_thresh=my_args.freq_thresh)[0]
        verbs_even = data_selector.subsample_even_score_distr(verbs_sorted, datapoints_per_bin, bin_size)
        print('range verb selection')
        print(f'score whole range count: {len(verbs_even)} \n Range from {min([score for word, score in verbs_even])} to {max([score for word, score in verbs_even])} \n{verbs_even}')
        vocab = verbs_even
    else:
        print('Vocab type not implemented. Choose between "mmn" (min, max, neutral), "mm" (min, max) or range (select even verbs from whole range).')
        sys.exit(-1)


    if not pca_exists:
        # ------------------------------------
        # ------------------------------------
        #           Else create pca
        # ------------------------------------
        # ------------------------------------
        print('Creating PCA from scratch...')

        random.shuffle(vocab)
        print(f'overall vocabulary length: {len(vocab)}')
        vocab_sentences = [item[0] for item in vocab] # get sentences
        vocab_labels = np.array([item[1] for item in vocab]).reshape(-1, 1) # get the labels

        # ------------------------------------
        #   Do PCA with the vocab embeddings
        # ------------------------------------
        print('------------------ Start PCA ------------------')
        dim_pca = DisDimPCA(n_components=my_args.dim, task_name=data_args.task_name, tensorboard_writer=tensorboard_writer, model_name=str(ID), device=device)

        # get the sentence embeddings of the vocabulary
        vocab_embeddings = dim_pca.sent_model.get_sen_embedding(vocab_sentences)
        if my_args.use_question_template:
            vocab_embeddings = dim_pca.sent_model.get_question_template_mean_sen_embeddings(vocab_sentences)

        dim_pca.fit(vocab_embeddings)

        # store pca as pickle file, if possible
        try:
            pickle.dump(dim_pca, open(pca_file_path,"wb"))
        except:
            print(f'Could not store dis dim pca to {pca_file_path}')

    if return_vocab:
        return dim_pca, vocab

    return dim_pca

def evaluate_pca(my_args, data_args, dim_pca, vocab, data_selector=None, plot_dir='plots/'):

    # add '/' to plot dir if not already in 
    if '/' not in plot_dir[-1:]: 
        plot_dir = plot_dir + '/'
    # check if plot dir exists, if not create dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if data_selector is None:
        data_selector = DataSelector()

    print(f'overall vocabulary length: {len(vocab)}')
    scatter_vocab(vocab, f'{my_args.vocab_type}_{my_args.vocab_center_strategy}', plot_dir=plot_dir)
    
    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=data_args.data_dir)

    if data_args.task_name == 'distress':
        lexicon = distress_lex
    else:
        lexicon = empathy_lex

    scatter_vocab(vocab, f'{my_args.vocab_type}_{my_args.vocab_center_strategy}', plot_dir=plot_dir)
    vocab_sentences = [item[0] for item in vocab] # get sentences
    vocab_labels = np.array([item[1] for item in vocab]).reshape(-1, 1) # get the labels
    vocab_embeddings = dim_pca.sent_model.get_sen_embedding(vocab_sentences)

    # get eigenvectors from best / highest eigenvalue
    transformed_emb = dim_pca.transform(vocab_embeddings)
    eigen_vec = dim_pca.pca.components_
    projection_highest_var = eigen_vec[0]

    dim_pca.scatter_vocab_words(vocab, transformed_emb[:, 0].reshape(-1), title_add_on=f'_{data_args.task_name}_random_y_dimension', plot_dir=plot_dir)
    dim_pca.scatter_vocab_words(vocab, transformed_emb[:, 0].reshape(-1), title_add_on=f'_{data_args.task_name}_dimension', set_y_random=False, plot_dir=plot_dir)

    # ------------------------------
    #    Analyse the PCA outcome
    # ------------------------------

    # --- Get principal component / Eigenvector ---
    # - How much variance (std..) do they cover? -
    # - how many are there? -
    var = dim_pca.explained_var
    print(list(var))
    pca_dim = transformed_emb[:, 0]

    plt.scatter(pca_dim, vocab_labels)
    plt.ylabel(f'{data_args.task_name} score')
    plt.xlabel('PCA dim')
    plt.title(f'PC1 covering {var[0]}')
    plt.savefig(f'EmpDim/{plot_dir}PCA_dim.pdf')
    plt.close()
    
    # plot cumsum
    plt.plot(var.cumsum())
    plt.xticks(list(range(len(var))))
    plt.savefig(f'EmpDim/{plot_dir}PCA_var_cumsum.pdf')    
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
    note = 'all_words_rand'
    all_words_rand_embeddings = dim_pca.sent_model.get_sen_embedding(all_words_rand)
    r_rand, p_rand = dim_pca.correlate_dis_dim_scores(all_words_rand_embeddings, all_words_rand_labels, printing=True)
    dim_pca.plot_dis_dim_scores(all_words_rand_embeddings, all_words_rand_labels, r_rand, title_add_on=note, plot_dir=plot_dir)
    dim_pca.update_log(my_args,
                data_args.task_name,
                pearson_r=r_rand, 
                pearson_p=p_rand, 
                vocab=vocab,
                center=data_selector.vocab_center,
                note=note,
                data_len=len(all_words_rand_embeddings),
                )

    # --- correlate for evenly sampled data samples ---
    
    datapoints_per_bin = 15
    print(f'correlate for even subsamples {datapoints_per_bin}')
    even_subsamples = data_selector.subsample_even_score_distr(all_words_n_scores, datapoints_per_bin=datapoints_per_bin, bin_size=0.1)
    sentences_input = [item[0] for item in even_subsamples]
    embedding_input = dim_pca.sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    note = 'all_words_even_15'
    true_labels = [item[1] for item in even_subsamples]
    r_even_15, p_even_15 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_even_15, title_add_on=note)
    dim_pca.update_log(my_args, 
                data_args.task_name,
                pearson_r=r_even_15, 
                pearson_p=p_even_15, 
                vocab=vocab,
                center=data_selector.vocab_center,
                note=note,
                data_len=len(sentences_input),
                )

                
    # --- correlate for evenly sampled data samples ---
    """
    datapoints_per_bin = 20
    print(f'correlate for even subsamples {datapoints_per_bin}')
    even_subsamples = data_selector.subsample_even_score_distr(all_words_n_scores, datapoints_per_bin=datapoints_per_bin, bin_size=0.1)
    sentences_input = [item[0] for item in even_subsamples]
    embedding_input = sent_model.get_sen_embedding(sentences_input)
    print('Dataset size:', len(embedding_input))
    note = 'all_words_even_20'
    true_labels = [item[1] for item in even_subsamples]
    r_even_20, p_even_20 = dim_pca.correlate_dis_dim_scores(embedding_input, true_labels, printing=True)
    dim_pca.plot_dis_dim_scores(embedding_input, true_labels, r_even_20, title_add_on=note)
    dim_pca.update_log(my_args, 
                data_args.task_name,
                pearson_r=r_even_20, 
                pearson_p=p_even_20, 
                vocab=vocab,
                center=data_selector.vocab_center,
                note=note,
                data_len=len(sentences_input),
                )

    """
    """
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
    

    if my_args.store_run: save_run(dim_pca.logging)
    """
    # ------------------------------
    #    Apply PCA to the essays
    # ------------------------------

    print('\n Apply PCA to the essays \n')
    # --- preprocess data ---
    # - load data -

    data_train_pd, data_dev_pd, data_test_pd = utils.load_data_complete(train_file=data_args.train_file, dev_file=data_args.validation_file, dev_label_file=data_args.validation_labels_file, test_file=data_args.test_file)
    data_train_pd = utils.clean_raw_data(data_train_pd, keep_id=True)
    data_dev_pd = utils.clean_raw_data(data_dev_pd, keep_id=True)
    data_test_pd = utils.clean_raw_data(data_test_pd, keep_id=True)

    data_train_pd = data_train_pd[:200]

    train_labels = np.array(data_train_pd[data_args.task_name]).reshape(-1, 1)
    train_sent = dim_pca.sent_model.get_sen_embedding(data_train_pd['essay'])
    train_sent, train_labels = shuffle(train_sent, train_labels, random_state=data_args.data_seed)
    train_sent_transformed = dim_pca.transform(train_sent)  

    for i in range(train_sent_transformed.shape[1]):
        #print(all_word_dataset_transformed.shape[1])
        princ_comp = i
        print(f'principal component {princ_comp}')
        # version 1:
        train_sent_transformed_i = train_sent_transformed[:, i]  # same result as multiplying data with the one principal component
        r, p = pearsonr(train_sent_transformed_i, train_labels)
        print('r', r)
        print('p', p)


    # generate labels for test set
    try:
        test_sent = dim_pca.sent_model.get_sen_embedding(data_test_pd['essay'])
        test_sent_transformed = dim_pca.transform(test_sent)
        df = pd.DataFrame(test_sent_transformed)
        output_dir = this_file_path + '/../output'
        output_dir = output_dir + f'/EmpDim/{data_args.task_name}/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = output_dir + f'pca_dim_test_{data_args.task_name}.tsv'
        df.to_csv(output_path, index=None, sep='\t')
    except Exception as e:
        print(f'No output generated: {e}. Outputdir: {output_dir}')
        pass # doesn't really matter here

    

           
    # ---
    # use other sentence mbeddings





    # - encode data -
    # - tranform data -
    # --- analyse data ---
    # - correlate this score with the actual label -



def run():
    """Running and evaluating pca
    """
    
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")


    # get arguments
    parser = HfArgumentParser((PCAArguments, DataTrainingArguments))
    my_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.task_name not in ['empathy', 'distress']:
        print("task name not available, choose 'empathy' or 'distress' next time. Usign empathy now")
        data_args.task_name = 'empathy'

    torch.manual_seed(data_args.data_seed)
    random.seed(data_args.data_seed)
    
    str_center_strategy = '_' + my_args.vocab_center_strategy if 'mmn' in my_args.vocab_type else ''
    tensorboard_writer = SummaryWriter(f'runs/{data_args.task_name}_{my_args.vocab_type}{str_center_strategy}{"_fdis" if my_args.use_freq_dist else ""}_{ID}{my_args.model_name}') if data_args.use_tensorboard else None

    data_selector = DataSelector()

    dim_pca, vocab = create_pca(my_args, data_args, tensorboard_writer=tensorboard_writer, return_vocab=True, data_selector=data_selector)
    evaluate_pca(my_args, data_args, dim_pca, vocab)


if __name__ == '__main__':
    run()