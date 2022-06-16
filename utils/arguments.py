# Outsource argument dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class PCAArguments:
    # --- organisationa settings for the run ---
    data_dir: str = field(
        default='data/', metadata={"help": "A directory containing the data."}
    )
    task_name: Optional[str] = field(
        default='distress',
        metadata={"help": "The task name to perform the model on. Either distress or empathy."},
    )
    model_name: Optional[str] = field(
        default='',
        metadata={"help": "The transformer model name for loading the tokenizer and pre-trained model."},
    )
    store_run: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, the run will be stored in json."},
    )
    use_tensorboard: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tensorboard will be used."},
    )
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
    seed: Optional[str] = field(
        default=17,
        metadata={"help": "The seed."},
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
    