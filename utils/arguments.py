# Outsource argument dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # edited by Myra Z.
    data_dir: str = field(
        default='data', metadata={"help": "A directory containing the data."}
    )
    # edited by Myra Z.
    task_name: Optional[str] = field(
        default='distress',
        metadata={"help": "The name of the task to train on"},
    )
    # edited by Myra Z.
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    # added by Myra Z.
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "The entity name of the wandb user. Leave empty if you do not wish to use wandb"}
    )
    # added by Myra Z.
    wandb_project: str = field(
        default="Results", metadata={"help": "The Project of the wandb."}
    )
    # added by Myra Z.
    use_tensorboard: Optional[bool] = field(
        default=False, metadata={"help": "If True, use a writer for tensorboard"}
    )
    # added by Myra Z.
    tensorboard_output_dir: str = field(
        default="runs/", metadata={"help": "Path to the sub directory of the writer. Saves in runs/ + output_dir"}
    )
    # edited by Myra Z.
    train_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/messages_train_ready_for_WS.tsv', metadata={"help": "A csv or a json file containing the training data."}
    )
    # edited by Myra Z.
    validation_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/messages_dev_features_ready_for_WS_2022.tsv', metadata={"help": "A csv or a json file containing the validation data."}
    )
    # edited by Myra Z.
    validation_labels_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/goldstandard_dev_2022.tsv', metadata={"help": "A csv or a json file containing the validation data."}
    )
    # edited by Myra Z.
    test_file: Optional[str] = field(default=data_dir.default + '/buechel_empathy/messages_test_features_ready_for_WS_2022.tsv', metadata={"help": "A csv or a json file containing the test data."})
    data_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "seed for selecting subset of the dataset if not using all."
        },
    )
    train_as_val: bool = field(
        default=False,
        metadata={"help": "if True, sample 1k from train as val"},
    )
    early_stopping_patience: Optional[int] = field(
        default=10,
    )    

    def __post_init__(self):
        # overwritten by Myra Z.
        if not os.path.exists(self.data_dir):  # Addition from Myra Zmarsly
            raise ValueError(f"The data directory: {self.data_dir} does not exists.")
        elif not os.listdir(self.data_dir):
            raise ValueError(f"The data directory {self.data_dir} is empty.")
        elif (not os.path.exists(self.train_file)) or (not os.path.exists(self.validation_file))or (not os.path.exists(self.validation_labels_file)):
            raise ValueError(f"The buechel_empathy data does not exist {self.data_dir} or is not stored / named corretly. The data should be in dir /buechel_empathy/")

@dataclass
class PCAArguments:
    # --- organisationa settings for the run ---
    #data_dir: str = field(  # TODO: Double
    #    default='data/', metadata={"help": "A directory containing the data."}
    #)
    #task_name: Optional[str] = field(  # TODO: Double
    #    default='distress',
    #    metadata={"help": "The task name to perform the model on. Either distress or empathy."},
    #)
    model_name: Optional[str] = field(
        default='',
        metadata={"help": "The transformer model name for loading the tokenizer and pre-trained model."},
    )
    store_run: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the run will be stored in json."},
    )
    #use_tensorboard: Optional[bool] = field(
    #    default=False,
    #    metadata={"help": "If True, tensorboard will be used."},
    #)
    run_id: str = field(
        default=None,
        metadata={"help": "If True, tensorboard will be used."},
    )
    # --- general training setting ---
    data_lim: Optional[int] = field(
        default=None,
        metadata={"help": "The data limit for the lexicon datadim."},
    )
    dim: Optional[int] = field(
        default=1,
        metadata={"help": "The n_components of the PCA / dimension."},
    )
    # --- set the vocabulary ---
    vocab_size: Optional[int] = field(
        default=10,
        metadata={"help": "The size of the vocabualry for max, min or neutral scores. If vocab_type = 'range' this int is the data size to select per bin."},
    )
    vocab_type: Optional[str] = field(
        default='mm',
        metadata={"help": "Available types are 'mm' (min max), 'mmn' (min max neutral), 'range' (use verbs from the whole range)."},
    )
    vocab_center_strategy: Optional[str] = field(
        default='soft',
        metadata={"help": "Available types are 'soft', 'hard'."},
    )
    vocab_bin_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "Will only be used when vocab_type = 'range'. The bin size for the vocabulary to select from."},
    )
    use_freq_dist: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the frequency distriution of words will filter the vocab. Based on threshold set: freq_thresh"},
    )
    freq_thresh: Optional[float] = field(
        default=0.00002,
        metadata={"help": "If True, the frequency distriution of words will filter the vocab. Based on threshold set: freq_thresh"},
    )
    random_vocab: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the vocabulary will be chosen random."},
    )
    use_question_template: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the template of a question will be used before creating the sentence embeddings."},
    )
    
