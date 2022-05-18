from transformers import AutoTokenizer, AutoModel
from transformers import HfArgumentParser
from datasets import Dataset

import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import os
import sys
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import utils
import preprocessing

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


def run():
    # get arguments
    parser = HfArgumentParser(MyArguments)
    my_args = parser.parse_args_into_dataclasses()[0]

    torch.manual_seed(my_args.seed)

    # ------------------------
    #     Load the lexicon 
    # ------------------------
    empathy_lex, distress_lex = utils.load_empathy_distress_lexicon(data_root_folder=my_args.data_dir)

    # --- select the most representative words for low and high distress and empathy ---
    # n words with highest and words with lowest ranking values
    data_count = 20  # datacount per min / max
    emp_min, emp_max = select_vocab(empathy_lex, data_count)
    dis_min, dis_max = select_vocab(distress_lex, data_count)

    # --- create correct data shape ---
    # create Huggingface dataset
    def create_dataset(data_input):
        data_dict = {'word': [item[0] for item in data_input], 'label': [item[1] for item in data_input]}
        return Dataset.from_dict(data_dict)

    emp_ds = create_dataset(emp_min + emp_max)
    dis_ds = create_dataset(dis_min + dis_max)

    # --- Optional: Create sentence with those words or context in general ---
    # I don't know what it would look like in our case

    # ----------------------------------------------
    #    Get the bert embeddings of those words 
    # ----------------------------------------------
    # - load tokenizer and encode words -
    tokenizer = AutoTokenizer.from_pretrained(my_args.model_name)

    emp_encoded = emp_ds.map(lambda x: preprocessing.tokenize(x, tokenizer, 'word', max_length=16), batched=True, batch_size=None)
    dis_encoded = dis_ds.map(lambda x: preprocessing.tokenize(x, tokenizer, 'word', max_length=16), batched=True, batch_size=None)

    # --- shuffle data ---
    emp_encoded_shuff = emp_encoded.shuffle(seed=my_args.seed)
    dis_encoded_shuff = dis_encoded.shuffle(seed=my_args.seed)

    # get input_ids, attention_mask and labels as numpy arrays and cast types
    # empathy
    input_ids_emp = np.array(emp_encoded_shuff["input_ids"]).astype(int)
    attention_mask_emp = np.array(emp_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_emp = np.array(emp_encoded_shuff["label"]).astype(np.float32).reshape(-1, 1)
    # distress
    input_ids_dis = np.array(emp_encoded_shuff["input_ids"]).astype(int)
    attention_mask_dis = np.array(emp_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_dis = np.array(emp_encoded_shuff["label"]).astype(np.float32).reshape(-1, 1)

    # get the bert embeddings of those words
    model = AutoModel.from_pretrained(my_args.model_name)
    # TODO: Get embeddings here
    

    # ------------------------------
    #   Do PCA with the embeddings
    # ------------------------------


    # ------------------------------
    #    Analyse the PCA outcome
    # ------------------------------
    # --- Get principal component / Eigenvector ---
    # - How much variance (std..) do they cover? -
    # - how many are there?



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