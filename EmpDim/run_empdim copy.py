from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser
from datasets import Dataset

import torch

from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import os
import sys
import os
import sys
from funcs_mcm import BERTSentence
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from torch.utils.data import DataLoader
import utils.utils as utils
import utils.preprocessing as preprocessing
import matplotlib.pyplot as plt

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
        default='bert-base-uncased',
        metadata={"help": "The transformer model name for loading the tokenizer and pre-trained model."},
    )
    seed: Optional[str] = field(
        default=17,
        metadata={"help": "The seed."},
    )
    tokenizer_len: Optional[str] = field(
        default=256,
        metadata={"help": "The seed."},
    )
    data_size: Optional[int] = field(
        default=20,
        metadata={"help": "The seed."},
    )


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

    torch.manual_seed(my_args.seed)

    # ------------------------
    #     Load the lexicon 
    # ------------------------
    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=my_args.data_dir)

    if my_args.task_name == 'distress':
        lexicon = distress_lex
    else:
        lexicon = empathy_lex

    # --- select the most representative words for low and high distress and empathy ---
    # n words with highest and words with lowest ranking values
    data_count = my_args.data_size # datacount per min / max
    score_min, score_max = select_vocab(lexicon, data_count)

    # --- create correct data shape ---
    # create Huggingface dataset
    def create_dataset(data_input):
        data_dict = {'word': [item[0] for item in data_input], 'label': [item[1] for item in data_input]}
        return Dataset.from_dict(data_dict)

    words_dataset = create_dataset(score_min + score_max)

    # --- Optional: Create sentence with those words or context in general ---
    # I don't know what it would look like in our case

    # ----------------------------------------------
    #    Get the bert embeddings of those words 
    # ----------------------------------------------
    # - load tokenizer and encode words -
    tokenizer = AutoTokenizer.from_pretrained(my_args.model_name)

    words_encoded = words_dataset.map(lambda x: preprocessing.tokenize(x, tokenizer, 'word', max_length=my_args.tokenizer_len), batched=True, batch_size=None)
    
    # --- shuffle data ---
    words_encoded_shuff = words_encoded.shuffle(seed=my_args.seed)

    # get input_ids, attention_mask and labels as numpy arrays and cast types
    # empathy
    print(words_encoded)
    input_ids = torch.tensor(np.array(words_encoded["input_ids"]).astype(int))
    attention_mask = torch.tensor(np.array(words_encoded["attention_mask"]).astype(int))
    labels = torch.tensor(np.array(words_encoded["label"]).astype(np.float32).reshape(-1, 1))

    # ------------------------
    # Sentence embeddings like MoRT: https://github.com/ml-research/MoRT_NMI/blob/master/MoRT/mort/funcs_mcm.py
    # \cite{Schramowski2021}
    """
    emb = BERTSentence(device=device, transormer_model='bert-large-nli-mean-tokens')
    get_sen_embedding_ = emb.get_sen_embedding


    #for i in tqdm(range(int(len(words_dataset) // n) + 1)):
    #    batch = words_dataset[i * n: i * n + n]
    #    res += 
    encoded = get_sen_embedding_(words_encoded['word'], dtype='list')
    print(encoded)
    return
    """
    sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.']

    sentences = words_encoded['word']

    #Sentences are encoded by calling model.encode()
    embeddings = sent_model.encode(sentences)

    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")



    # ------------------------

    # - get the bert embeddings of those words -
    model = AutoModel.from_pretrained(my_args.model_name, output_hidden_states = True)

    def get_mean_sent_emb(model, input_ids, attention_mask):

        # https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # Remove the first hidden state (input)
            hidden_states = outputs[2][1:]
        
        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        token_embeddings_np = token_embeddings.cpu().detach().numpy()
        return token_embeddings_np.mean(axis=1)

    token_embeddings_np = np.array(embeddings)  # use sentence embessings 

    # is simply taking the mean of all embeddings correct or do we need to ignore the ones that are 0?
    #sums = token_embeddings_np.sum(axis=1)
    #nonzero_counts = np.count_nonzero(attention_mask_emp, axis=1).reshape((-1, 1))
    # (sums / nonzero_counts)
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #print(X.shape)
    #test_pca = PCA()  # n_components=2
    #test_pca.fit(X)
    #print(test_pca.explained_variance_ratio_)

    # - compute the mean sentence embedding -
    #token_embeddings_np[token_embeddings_np == 0] = np.nan
    #mean_token_emb = np.nanmean(token_embeddings_np, axis=1)  # 
    mean_token_emb = get_mean_sent_emb(model, input_ids, attention_mask)
    print(mean_token_emb.shape)

    mean_token_emb = token_embeddings_np#.reshape(-1, 1) # use sentence embessings 
    print('new mean emb shape', mean_token_emb.shape)

    # ------------------------------
    #   Do PCA with the embeddings
    # ------------------------------
    pca = PCA()
    transformed_emb = pca.fit_transform(mean_token_emb)
    
    # ------------------------------
    #    Analyse the PCA outcome
    # ------------------------------

    # --- Get principal component / Eigenvector ---
    # - How much variance (std..) do they cover? -
    # - how many are there?
    var = pca.explained_variance_ratio_
    print(var)
    #for i in range(label_emp.shape[0]):
    #    print('\n', i)
    #    print('transformed_emb', transformed_emb[i])
    #    print('label_emp', label_emp[i])

    princ_comp = 0  # principal component: 0 means first
    if pca.explained_variance_ratio_.shape[0] > 1:
        pca_dim = transformed_emb[:, 0]
    else:
        pca_dim = transformed_emb

    plt.scatter(pca_dim, labels)
    plt.ylabel('empathy / distress label')
    plt.xlabel('PCA dim')
    plt.title(f'PC1 covering {var[princ_comp]}')
    plt.savefig('EmpDim/plots/PCA_dim.pdf')
    print('Done showing plot')
    plt.close()
    
    # plot cumsum
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xticks(list(range(len(pca.explained_variance_ratio_))))
    plt.savefig('EmpDim/plots/PCA_var_cumsum.pdf')
    plt.close()

    # ------------------------------
    #    Apply PCA to the essays
    # ------------------------------
    # TODO
    # --- preprocess data ---
    # - load data -
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=my_args.data_dir)
    data_train_pd = utils.clean_raw_data(data_train_pd[:50])
    data_dev_pd = utils.clean_raw_data(data_dev_pd[:50])

    # --- get fully preprocessed data ---
    dataset_emp_train, dataset_dis_train  = preprocessing.get_preprocessed_dataset_huggingface(data_train_pd, tokenizer, my_args.seed, return_huggingface_ds=False, padding='max_length', max_length=my_args.tokenizer_len)
    dataset_emp_dev, dataset_dis_dev = preprocessing.get_preprocessed_dataset_huggingface(data_dev_pd, tokenizer, my_args.seed, return_huggingface_ds=False, padding='max_length', max_length=my_args.tokenizer_len)
    
    # --- create dataloader ---
    # for empathy
    batch_size = 32
    dataloader_emp_train = DataLoader(dataset_emp_train, batch_size=batch_size, shuffle=True)
    dataloader_emp_dev = DataLoader(dataset_emp_dev, batch_size=batch_size, shuffle=True)
    # for distress
    dataloader_dis_train = DataLoader(dataset_dis_train, batch_size=batch_size, shuffle=True)
    dataloader_dis_dev = DataLoader(dataset_dis_dev, batch_size=batch_size, shuffle=True)

    # --- choose dataset and data loader based on empathy ---
    # per default use empathy label
    dataloader_train = dataloader_emp_train
    dataloader_dev = dataloader_emp_dev
    dataset_train = dataset_emp_train
    dataset_dev = dataset_emp_dev
    display_text = 'Using empathy data'
    if my_args.task_name == 'distress':
        dataset_train = dataset_dis_train  # needed for k fold
        dataset_dev = dataset_dis_dev  # needed for k fold
        dataloader_train = dataloader_dis_train
        dataloader_dev = dataloader_dis_dev
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')

    # essay encoded
    #print(dataset_train)
    #train_input_ids = torch.tensor(np.array(dataset_train["input_ids"]).astype(int))
    #train_attention_mask = torch.tensor(np.array(dataset_train["attention_mask"]).astype(int))
    #train_labels = torch.tensor(np.array(dataset_train["label"]).astype(np.float32).reshape(-1, 1))

    #transformed_train = []
    labels_from_batch = []
    for step, batch in enumerate(dataloader_train): 
    #    #print(dataloader_train[batch[0]])
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        labels_from_batch = labels_from_batch + batch_labels.cpu().detach().numpy().tolist()
    #    print(step * len(batch_inputs))
    #    #print(dataloader_train[batch[0]])
    #    batch_mean_embeddings = get_mean_sent_emb(model, batch_inputs, batch_masks)
    #    transformed_batch = pca.transform(batch_mean_embeddings).tolist()
    #    transformed_train = transformed_train + transformed_batch
    #transformed_train = np.array(transformed_train)
    labels_from_batch = np.array(labels_from_batch)

    # ---
    # use other sentence mbeddings
    sentences_train = data_train_pd['essay']

    #Sentences are encoded by calling model.encode()
    embeddings_train = sent_model.encode(sentences_train)
    transformed_train = pca.transform(embeddings_train)
    transformed_train = np.array(transformed_train)
    # ---

    #train_mean_embeddings = get_mean_sent_emb(model, train_input_ids, train_attention_mask)
    #transformed_train = pca.transform(train_mean_embeddings)

    princ_comp = 0  # principal component: 0 means first
    if pca.explained_variance_ratio_.shape[0] > 1:
        pca_dim_train = transformed_train[:, 0]
    else:
        pca_dim_train = transformed_train

    plt.scatter(pca_dim_train, labels_from_batch)
    plt.ylabel('empathy / distress label')
    plt.xlabel('PCA dim')
    plt.title(f'PC1 covering {var[princ_comp]}')
    plt.savefig('EmpDim/plots/PCA_dim_train.pdf')
    print('Done showing plot')
    plt.close()
    r, p = pearsonr(pca_dim_train, labels_from_batch)
    print('r', r)
    print('p', p)


    # - encode data -
    # - tranform data -
    # --- analyse data ---
    # - correlate this score with the actual label -





def select_vocab(lexicon, count):
    """Select the vocabulary with max and min ranking score in the dictionary

    Args:
        lexicon ({str: float}}): The lexicon holding the word with a ranking
        count (_type_): The number of datapoints to retrun per lexicon

    Returns:
        lex_min, lex_max: The sorted vocabularies
    """
    sorted_lex_max = sorted(lexicon.items(), key=lambda item: item[1], reverse=True)
    sorted_lex_min = sorted(lexicon.items(), key=lambda item: item[1])
    return sorted_lex_min[:count], sorted_lex_max[:count]


if __name__ == '__main__':
    run()