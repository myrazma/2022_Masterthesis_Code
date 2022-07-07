import pickle
from regex import F
import sklearn
from sklearn.decomposition import PCA

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from EmpDim.funcs_mcm import BERTSentence
import utils.utils as utils
import utils.preprocessing as preprocessing
from utils.arguments import PCAArguments, DataTrainingArguments

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

sent_model = BERTSentence(device=device)

# TODO, add mort file to DataTrainArgs
data_args = DataTrainingArguments()
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

# ------------------
# create moral score
# ------------------
print(train_dataset)
essays = train_dataset['essay'][:100]
labels = train_dataset['label'][:100]

essay_embeddings = sent_model.get_sen_embedding(essays)

mort_pca = load_mort_pca(filename=data_args.data_dir + '/MoRT_projection/projection_model.p')
moral_dim = mort_pca.transform(essay_embeddings)

print(type(moral_dim))
print(moral_dim)
try:
    print(moral_dim.size())
except:
    pass