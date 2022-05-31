from pickle import FALSE
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
    data_size: Optional[int] = field(
        default=20,
        metadata={"help": "The size of the vocabualry for max min scores."},
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
    


def stem_words(words):
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
    stemmer = PorterStemmer()
    if isinstance(words, list):
        if isinstance(words[0], tuple):
            word_stems = [(stemmer.stem(w), score) for (w, score) in words]
        elif isinstance(words[0], str):
            word_stems = [stemmer.stem(w) for w in words]
        else:
            print('MyWarning in stem_words(): variable "words" should be a list of strings or list of tuples. Returning empty list.')
            word_stems = []
    elif isinstance(words, str):
        word_stems = stemmer.stem(words)
    else:
        print('MyWarning in stem_words(): variable "words" should be a list of strings or list of tuples. Returning empty list.')
        word_stems = []
    return word_stems


def remove_dublicates(words, sorting='max'):
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


def get_verbs(words):
    """Get the words from a list

    Args:
        words (list((str, float)) or list(str)): The list of words

    Returns:
        list((str, float)) or list(str): The verbs
    """
    verbs = []
    blacklist = ['blanket', 'home', 'shipwreck', 'cub']
    blacklist = ['blanket', 'home', 'shipwreck', 'cub', 'joke', 'fart', 'gag', 'clown']
    if isinstance(words, list):
        if isinstance(words[0], tuple):
            for word, score in words:
                if word not in blacklist:
                    verb_synset_ls = wn.synsets(word, pos=wn.VERB)  # if a verb can be found in the list
                    if len(verb_synset_ls) >= 1:
                        #print(f'{word} is a verb (score: {score})\n Synset: {verb_synset_ls}\n')
                        verbs.append((word, score))
        if isinstance(words[0], str):
            for word in words:
                verb_synset_ls = wn.synsets(word, pos=wn.VERB)
                if len(verb_synset_ls) >= 1:
                    verbs.append(word)

    return verbs


def select_words(lexicon, word_count, random_vocab=False):
    # return min and max sorted words with length of word_count
    print(f'Vocabulary info: \n word count: {word_count}')
    def __get_words(sorting):
        reverse = True if sorting=='max' else False
        words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1], reverse=reverse)]
        word_stems = stem_words(words_sorted)
        distinct_word_stems = remove_dublicates(word_stems, sorting=sorting)
        distinct_verbs = get_verbs(distinct_word_stems)
        distinct_verbs_selected = distinct_verbs[:word_count]
        # shuffle for random
        if random_vocab:
            distinct_verbs_selected = distinct_verbs.copy()
            random.shuffle(distinct_verbs_selected)
            distinct_verbs_selected = distinct_verbs_selected[:word_count]
        print(f'Mode: {sorting}')
        print(f'Range from {min([score for word, score in distinct_verbs_selected])} to {max([score for word, score in distinct_verbs_selected])}')
        return distinct_verbs_selected
    words_min = __get_words(sorting='min')
    words_max = __get_words(sorting='max')
    return words_min, words_max


def save_run(my_args, vocab_min, vocab_max, pca_var, pca_pearsonr, pca_pearsonp, filename=None, tensorboard_writer=None):
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
    new_row_dict = {'id': START_TS,  # can also be trated as the individual id
                'dim': my_args.dim, 
                'task_name': my_args.task_name, 
                'data_lim': my_args.data_lim,
                'vocab_size': my_args.data_size,
                'random_vocab': my_args.random_vocab,
                'vocab_min': vocab_min_words,
                'vocab_min_scores': vocab_min_scores,
                'vocab_max': vocab_max_words,
                'vocab_max_scores': vocab_max_scores,
                'princip_comp': list(range(len(pca_pearsonr))),  # the number of the principal component
                'pca_var': pca_var,  # principal components
                'pca_pearsonr': pca_pearsonr,
                'pca_pearsonp': pca_pearsonp
                }

    #if tensorboard_writer is not None:
    #    with SummaryWriter() as w:
    #        for i in range(len(pca_pearsonr)):
    #            tensorboard_writer.add_hparams(new_row_dict, {'princip_comp': int(i), 'pca_var': float(pca_var[i]), 'pca_pearsonr': float(pca_pearsonr[i]), 'pca_pearsonp': pca_pearsonp[i]})
    new_row = pd.DataFrame(new_row_dict)
    df = pd.concat([df, new_row])
    print('Saved Information:\n', new_row[:5])
    
    df.to_csv(filename, sep=',')

def get_template():
    template = [
            ['Is it ok to {}?', 'No, it is not.', 'Yes, it is.'],
            ['To {} makes me feel.', 'No, you should not.', 'Yes, you should.'],
        ]


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

    score_range = (1, 7)
    # --- normalize the scores in the lexicon ---
    # lexicon = dict([(key, lexicon[key] / score_range[1]) for key in lexicon])

    # --- select the most representative words for low and high distress and empathy ---
    # n words with highest and words with lowest ranking values

    score_min, score_max = select_words(lexicon, my_args.data_size, random_vocab=my_args.random_vocab)

    # --- create correct data shape for min max data set---
    # create Huggingface dataset
    def create_dataset(data_input, shuffle=False, seed=17):
        """create dataset from list of tuples: list((word, label))

        Args:
            data_input (list((str, float))): _description_

        Returns:
            _type_: _description_
        """
        data_dict = {'word': [item[0] for item in data_input], 'label': [item[1] for item in data_input]}
        dataset = Dataset.from_dict(data_dict)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        return dataset

    print('\nWords with min score:\n', score_min)
    print('\nWords with max score:\n', score_max)
    min_max_words_dataset = create_dataset(score_min + score_max, shuffle=True, seed=my_args.seed)
    # create dataset for all words

    # with stemming of whole lexicon
    #all_words_stemmed = [(key, lexicon[key]) for key in lexicon]
    #all_words_stemmed = stem_words(all_words_stemmed)
    #all_word_dataset = create_dataset(all_words_stemmed[:my_args.data_lim])
    # altneative wihtout stemming:
    all_word_dataset = create_dataset([(key, lexicon[key]) for key in lexicon][:my_args.data_lim], shuffle=True, seed=my_args.seed)
    print('len(all_word_dataset)', len(all_word_dataset))

    sentences = min_max_words_dataset['word']  # get sentences
    labels = np.array(min_max_words_dataset['label']).reshape(-1, 1) # get the labels

    # ------------------------
    # Use Sentence embeddings like MoRT (using their code and functions in funcs_mcm.py): https://github.com/ml-research/MoRT_NMI/blob/master/MoRT/mort/funcs_mcm.py
    # Schramowski et al., 2021, Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do

    sent_model = BERTSentence(device=device) #, transormer_model='paraphrase-MiniLM-L6-v2') # TODO use initial model (remove transformer model varibale from head)
    sent_embeddings = sent_model.get_sen_embedding(sentences)

    # get sentence embeddings for all words dataset
    all_sentences = all_word_dataset['word']
    all_words_embeddings = sent_model.get_sen_embedding(all_sentences)
    all_words_labels = np.array(all_word_dataset['label']).reshape(-1, 1)



    #Print the embeddings
    #for sentence, embedding in zip(sentences, sent_embeddings):
    #    print("Sentence:", sentence)
    #    print("Embedding:", embedding)
    #    print("")

    # ------------------------------
    #   Do PCA with the embeddings
    # ------------------------------
    print('Do PCA')
    pca = PCA(n_components=my_args.dim)
    transformed_emb = pca.fit_transform(sent_embeddings)

    princ_comp_idx = 0  # principal component: 0 means first

    # get eigenvalues
    eigen_val = pca.explained_variance_ratio_

    # get eigenvectors from best / highest eigenvalue
    eigen_vec = pca.components_
    projection = eigen_vec[princ_comp_idx]  # TODO check if this line is really correct, am I selecting and getting the right values?
    print(projection.shape)
    # ------------------------------
    #    Analyse the PCA outcome
    # ------------------------------

    # --- Get principal component / Eigenvector ---
    # - How much variance (std..) do they cover? -
    # - how many are there?
    var = pca.explained_variance_ratio_
    print(list(var))
    if pca.explained_variance_ratio_.shape[0] > 1:
        pca_dim = transformed_emb[:, 0]
    else:
        pca_dim = transformed_emb[:]
    pca_dim = transformed_emb[:, 0]

    plt.scatter(pca_dim, labels)
    plt.ylabel('empathy / distress label')
    plt.xlabel('PCA dim')
    plt.title(f'PC1 covering {var[princ_comp_idx]}')
    plt.savefig('EmpDim/plots/PCA_dim.pdf')
    plt.close()
    
    # plot cumsum
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xticks(list(range(len(pca.explained_variance_ratio_))))
    plt.savefig('EmpDim/plots/PCA_var_cumsum.pdf')
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Scatter_Predictions', plt.gcf())

    plt.close()

    # --- check projection on all words datatset (all words from lexicon) ---
    # transform lexicon with pca

    #all_words_embeddings = all_words_embeddings[:data_restrict]
    #all_words_labels = all_words_labels[:data_restrict]
    def correlate_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=False, title_note=''):
        sentence_emb = sent_model.get_sen_embedding(sentences_input)
        sent_transformed = pca.transform(sentence_emb)
        true_labels = np.array(true_labels)

        pca_pearsonr, pca_pearsonp = [], []
        for i in range(sent_transformed.shape[1]):
            princ_comp = i
            print(f'principal component {princ_comp}')
            # version 1:
            sent_transformed_i = sent_transformed[:, i]  # same result as multiplying data with the one principal component
            r, p = pearsonr(sent_transformed_i, true_labels)
            if isinstance(r, list):
                r = r[0]
            print('r', r)
            print('p', p)
            pca_pearsonr.append(float(r))
            pca_pearsonp.append(float(p))

            # version 2: Same result as above
            #eigen_vec_i = pca.components_[princ_comp] 
            #all_word_dataset_transformed_i = all_words_embeddings @ eigen_vec_i
            #r, p = pearsonr(all_word_dataset_transformed_i, true_labels)

            filename = f'PCA_{my_args.task_name}_dim{my_args.dim}_PC{princ_comp}_vocab{my_args.data_size}{title_note}'
            title = f'PCA using {my_args.dim} dim(s), explaining {var[int(i)]} of var. Vocab size: {my_args.data_size}. \n pearson r: {r} \n task: {my_args.task_name}'
            plt.scatter(sent_transformed_i, true_labels)
            plt.ylabel('empathy / distress label')
            plt.xlabel('PCA dim for the whole lexicon')
            plt.title(title)
            if my_args.store_run: plt.savefig(f'EmpDim/plots/{filename}.pdf')
            if tensorboard_writer is not None:
                tensorboard_writer.add_figure(f'Scatter Predictions - PC {princ_comp}', plt.gcf())
            plt.close()

        if my_args.store_run:
            save_run(my_args, score_min, score_max, pca.explained_variance_ratio_, pca_pearsonr, pca_pearsonp, tensorboard_writer=tensorboard_writer)


    def correlate_combined_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=False):
        sentence_emb = sent_model.get_sen_embedding(sentences_input)
        sent_transformed = pca.transform(sentence_emb)
        true_labels = np.array(true_labels)

        if my_args.dim > 1:
            def normalize_scores(val):
                y = (val - min(val))
                y  = y / max(y)
                return y
                
            #labels = normalize_scores(true_labels)  # normalize labels
            labels = true_labels.reshape((-1,))
            weights = np.ones(sent_transformed.shape[1]).reshape((-1,1))

            y_pred = sent_transformed @ weights
            r, p = pearsonr(labels, y_pred)
            if isinstance(r, list):
                r = r[0]
            print('Combined pearson r:', r)
            print('Combined pearson p:', p)
            
            filename = f'PCA_{my_args.task_name}_dim{my_args.dim}_vocab{my_args.data_size}_combined_PCs'
            title = f'PCA using {my_args.dim} dim(s), explaining {var.cumsum()[int(my_args.dim - 1)]} of var. Vocab size: {my_args.data_size}. \n pearson r: {r} \n task: {my_args.task_name}'
            plt.scatter(y_pred, labels)
            plt.ylabel('empathy / distress label')
            plt.xlabel('PCA dim for the whole lexicon')
            plt.title(title)
            if store_run: plt.savefig(f'EmpDim/plots/{filename}.pdf')
            if tensorboard_writer is not None:
                tensorboard_writer.add_figure('Scatter Predictions - Combine all dimensions', plt.gcf())
            plt.close()

        if my_args.store_run:
            save_run(my_args, score_min, score_max, pca.explained_variance_ratio_, [r], [p], tensorboard_writer=tensorboard_writer)
        
    
    # overwrite sentences with not whole lexicon

    sentences_input = all_sentences
    true_labels = all_words_labels
    correlate_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=my_args.store_run)
    #correlate_combined_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=my_args.store_run)

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
    true_labels = [item[1] for item in words_min + words_max + words_random]
    correlate_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=my_args.store_run, title_note=f'_twotailed_{lim*2}_random{lim}')

    # without random

    print(f'\n Do correlation on the words with {lim} min an max scores (insg. {lim*2} words)')
    words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
    words_min = words_sorted[:lim]
    words_max = words_sorted[-lim:]
    sentences_input = [item[0] for item in words_min + words_max]
    true_labels = [item[1] for item in words_min + words_max]
    correlate_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=my_args.store_run, title_note=f'_twotailed_{lim*2}')

    #correlate_combined_emp_dim_pca(pca, sent_model, sentences_input, true_labels, store_run=my_args.store_run)




    # -----------------
    #   Empathy Bias
    # -----------------
    def correlate_emp_bias(bias_input, true_labels, norm_labels=False):
        true_labels = np.array(true_labels)
        score_range = (1, 7)
        if norm_labels:
            true_labels = (np.array(true_labels) - float(score_range[1] + 1)/2 ) / ((float(score_range[1]) - float(score_range[0]))/2)

        bias_res = sent_model.bias(bias_input)
        bias_labels = [item[0] for item in bias_res]

        i = 0
        for bias, word, score in zip(bias_labels, bias_input, true_labels):
            if i == 10:  # only print the first 10 words
                break
            print(f'Word: {word}. \n True Label --> Bias \n {score} --> {bias}')
            i += 1

        r, p = pearsonr(true_labels.reshape((-1,)), bias_labels)
        print(f'Pearson correlation of bias and true score. \n r: {r}, p: {p}\n')
        return r, p

    # --- do for all words in dictionary ---
    if False:

        lim = 100
        print(f'\n Do correlation on the words with {lim} min an max scores (insg. {lim*2} words)')
        # get min 100 and max 100 of the words
        # with random
        
        words_sorted = [(word, score) for word, score in sorted(lexicon.items(), key=lambda item: item[1])]
        words_min = words_sorted[:lim]
        words_max = words_sorted[-lim:]
        sentences_input = [item[0] for item in words_min + words_max]
        true_labels = [item[1] for item in words_min + words_max]
        
        print('Bias: two tailed sentences') # print('all_sentences')
        bias_input = sentences_input # all_sentences  # [item[0] for item in score_max]
        true_labels = true_labels # all_words_labels
        r, p = correlate_emp_bias(bias_input, true_labels, norm_labels=False)

        # --- do only for only verbs in dictionary ---
        # - random 200 verbs -
        selected_verbs, _  = select_words(lexicon, 100, random_vocab=True)
        print('200 selected random verbs')
        bias_input = [item[0] for item in selected_verbs]
        true_labels = [item[1] for item in selected_verbs]
        r, p = correlate_emp_bias(bias_input, true_labels, norm_labels=False)

        # --- Do for score min and score max ---
        print('\n Score min and score max')
        bias_input = [item[0] for item in score_min + score_max]
        true_labels = [item[1] for item in score_min + score_max]
        r, p = correlate_emp_bias(bias_input, true_labels, norm_labels=False)


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