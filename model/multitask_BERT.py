# ---------- Sources ----------
#
# [1] How to Create and Train a Multi-Task Transformer Model: 
#   https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
#   Use for Layering of Multi-Task learning with Pytorch and Huggingface Transformers
#   github code: https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
#
# ------------------------------

# utils
import logging
from logging import root
from pathlib import Path
import os
import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
# Transformers, torch and model utils
import transformers
from transformers import BertModel, BertConfig, BertForSequenceClassification, AutoModel
from transformers import BertTokenizer, AutoTokenizer, DataCollatorForTokenClassification
from transformers import AdamW, EvalPrediction, Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup, set_seed
import datasets
from datasets import Dataset, DatasetDict
from datasets import ClassLabel, load_dataset, load_metric
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from transformers.trainer_utils import get_last_checkpoint

# import own module
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils

logger = logging.getLogger(__name__)

class BertMultiTask(nn.Module):
    def __init__(self, tasks, bert_type='bert-base-uncased'):
        """The Multi task BERT model
        source: [1]

        Args:
            tasks (list(str)): A list of tasks
            bert_type (str, optional): The pre-trained Bert model type. Defaults to 'bert-base-uncased'.
        """
        super(BertMultiTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(bert_type)

        # create Module dict that holds the different heads
        self.output_heads = nn.ModuleDict()
        # create output heads for the single learning tasks
        for task in tasks:
            # init decoder with own defined SequenceRegressionHead
            decoder = SequenceRegressionHead(self.encoder.config.hidden_size, task.num_labels)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    def forward_own(self, input_ids, attention_masks, lexical_features):
        pass

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
    # source: [1]
    # added own description / comments for code understanding

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        regr_output = None
        for unique_task_id in unique_task_ids_list:

            # for the elements in batch, make True-False list to use as filter for selection
            # [task1, task2, task2] will hence be [False, True, True] is usique task ID is task2
            task_id_filter = task_ids == unique_task_id
            # in this step we will select only the relevant elements for the specific task head
            # and select the head from the output_head dictionary
            regr_output, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (regr_output, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs


class SequenceRegressionHead(nn.Module):
    # followed by tutorial from [1], adjusted to regression task
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.regressor = nn.Linear(hidden_size, num_labels)
        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)  # Dropout layer
        regr_output = self.regressor(pooled_output)  # Regression layer

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_mse = nn.MSELoss()
            loss = loss_mse(regr_output.view(-1, self.num_labels), labels.long().view(-1))

        return regr_output, loss

    def _init_weights(self):
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        if self.regressor.bias is not None:
            self.regressor.bias.data.zero_()


class Task():
    """A class for a task holding different informations about this task
    source [1]
    """
    def __init__(self, name, num_labels, id, output_type):
        """The initializer for the specific tasks

        Args:
            name (_type_): _description_
            num_labels (_type_): _description_
            id (_type_): _description_
            output_type (_type_): _description_
        """
        self.name = name
        self.num_labels = num_labels
        self.id = id
        self.output_type = output_type


def create_dataloaders(data_pd, batch_size):
    # Source: [2]
    dataset = Dataset.from_pandas(data_pd)
    #input_tensor = torch.tensor(inputs)
    #mask_tensor = torch.tensor(masks)
    #labels_tensor = torch.tensor(labels)
    #dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def compute_metrics(p: EvalPrediction):
    # followed by tutorial from [1], adjusted to regression task
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if preds.ndim == 1:
        # Sequence regression task, use correlation
        r_val, p_val = pearsonr(preds, p.label_ids)
        return {"corr": r_val}
    else:
        raise NotImplementedError()


def tokenize_pd(batch, tokenizer):
  # Source: [1] - https://huggingface.co/docs/transformers/training
  return pd.Series(dict(tokenizer(batch, padding='max_length', truncation=True, max_length=512)))


def tokenize(batch, tokenizer):
    # Source: [1] - https://huggingface.co/docs/transformers/training
    return tokenizer(batch['essay'], padding='max_length', truncation=True, max_length=512)


def create_dataloaders_pd(data_pd, batch_size):
    dataset = Dataset.from_pandas(data_pd)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_dataloaders(inputs, masks, labels, tasks, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    tasks_tensor = torch.tensor(tasks)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, tasks_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def run_old(root_folder="", empathy_type='empathy'):
    # for steps see here: https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240 
    # --------------------------------
    #           Load Data
    # --------------------------------

    # iIf we are adding tasks with different labels sizes (fpr empathy they have the dimension of 1), we should align all labels to the biggest vector
    # Load dataset with labels and features

    # make sure it is cleaned

    # Preprocessing: tokenizing, etc.

    # --------------------------------
    #   Create Multitasking Dataset
    # --------------------------------

    # combine dataset with sampled from both tasks, best doing while still pandas
    # TODO: Should we include one sample two times with two labels or on sample only with one label?

    # store which label is from what task, so that we can hand the different data points for teh tasks later on

    # shuffle: single batch should contain samples from multiple tasks

    # creating dataset and dataloader from huggingface
    pass



def run(params):

    # set training arguments
    training_args = TrainingArguments(
        do_train=params['do_train'],
        do_eval=params['do_eval'],
        output_dir=params['root_dir'] + params['output_dir'],
        learning_rate=params['learning_rate'],
        num_train_epochs=params['epochs'],
        overwrite_output_dir=True,
        seed=params['seed'],
    )
    batch_size = params['batch_size']
    # --------------------------------
    #          Load Data
    #          Source [1]
    # --------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    set_seed(training_args.seed)




    # --------------------------------
    #          Load Data
    # --------------------------------

    # iIf we are adding tasks with different labels sizes (fpr empathy they have the dimension of 1), we should align all labels to the biggest vector
    # Load dataset with labels and features
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=params['root_dir'] + "data/")

    # make sure it is cleaned
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)

    # only select necessary columns
    data_train_pd = data_train_pd[['distress', 'empathy', 'essay']]
    data_dev_pd = data_dev_pd[['distress', 'empathy', 'essay']]

    # as transformers dtaaset
    data_train = Dataset.from_pandas(data_train_pd)
    data_dev = Dataset.from_pandas(data_dev_pd)

    # --------------------------------
    #          Preprocessing
    # --------------------------------

    # tokenizing
    tokenizer = AutoTokenizer.from_pretrained(params['bert_type'])

    data_train_encoded = data_train.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)

    # --- shuffle data ---
    data_train_encoded_shuff = data_train_encoded.shuffle(seed=params['seed'])
    data_dev_encoded_shuff = data_dev_encoded.shuffle(seed=params['seed'])

    # get input_ids, attention_mask and labels
    # train
    input_ids_train = np.array(data_train_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_train_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_train = np.array(data_train_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_train = np.array(data_train_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    # dev
    input_ids_dev = np.array(data_dev_encoded_shuff["input_ids"]).astype(int)
    attention_mask_dev = np.array(data_dev_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_dev = np.array(data_dev_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_dev = np.array(data_dev_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)

    # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
    label_scaled_empathy_train = utils.normalize_scores(label_empathy_train, (1,7))
    label_scaled_empathy_dev = utils.normalize_scores(label_empathy_dev, (1,7))
    label_scaled_distress_train = utils.normalize_scores(label_distress_train, (1,7))
    label_scaled_distress_dev = utils.normalize_scores(label_distress_dev, (1,7))

    # create array for tasks
    empathy_task = Task(name='empathy', num_labels=1, id=0, output_type='regression')
    distress_task = Task(name='distress', num_labels=1, id=1, output_type='regression')
    task_list = [empathy_task, distress_task]
    task_empathy_train = np.array([empathy_task.id for i in range(label_scaled_empathy_train.shape[0])]).reshape(-1, 1)
    task_distress_train = np.array([distress_task.id for i in range(label_scaled_distress_train.shape[0])]).reshape(-1, 1)
    task_empathy_dev = np.array([empathy_task.id for i in range(label_scaled_empathy_dev.shape[0])]).reshape(-1, 1)
    task_distress_dev = np.array([distress_task.id for i in range(label_scaled_distress_dev.shape[0])]).reshape(-1, 1)

    # combine data into multiinput data, every data point two times (one time for empathy, one time for distress)
    # train
    multi_input_ids_train = np.vstack((input_ids_train, input_ids_train))  # stack the input ids two times on top
    multi_attention_mask_train = np.vstack((attention_mask_train, attention_mask_train))  # stack the attention masks two times on top
    multi_label_train = np.vstack((label_scaled_empathy_train, label_scaled_distress_train))  # stack the label for empathy and distress on top of each other
    multi_tasklabel_train = np.vstack((task_empathy_train, task_distress_train))  # additionally stack task class to identify type of classification later on

    # dev
    multi_input_ids_dev = np.vstack((input_ids_dev, input_ids_dev))  # stack the input ids two times on top
    multi_attention_mask_dev = np.vstack((attention_mask_dev, attention_mask_dev))  # stack the attention masks two times on top
    multi_label_dev = np.vstack((label_scaled_empathy_dev, label_scaled_distress_dev))  # stack the label for empathy and distress on top of each other
    multi_tasklabel_dev = np.vstack((task_empathy_dev, task_distress_dev))  # additionally stack task class to identify type of classification later on

    # --- create dataloader ---
    multi_dataloader_train = create_dataloaders(multi_input_ids_train, multi_attention_mask_train, multi_label_train, multi_tasklabel_train, batch_size)
    multi_dataloader_dev = create_dataloaders(multi_input_ids_dev, multi_attention_mask_dev, multi_label_dev, multi_tasklabel_dev, batch_size)

    for batch in multi_dataloader_dev:
        input_ids, attention_mas, labels, tasks = batch
        print(input_ids)
        print(attention_mas)
        print(labels)
        print(tasks)
        break
    #------ hhhh ------

    # normalize / scale data with minmaxscaler from [1,7] to [0,1] 
    
    # create dataset for each task
    # empathy
   
    # store which label is from what task, so that we can hand the different data points for teh tasks later on
 
    # rename individual target name to label
 
    # shuffle
 
    # --------------------------------
    #   Create Multitasking Dataset
    # --------------------------------

    # combine dataset with samples from both tasks, best doing while still pandas
    # shuffle: single batch should contain samples from multiple tasks
    # TODO: Should we include one sample two times with two labels or on sample only with one label?
  
    # creating dataset and dataloader from huggingface
    return

    # ---------------------------------------------------------------
    # delete start

    # tokenizing
    tokenizer = AutoTokenizer.from_pretrained(params['bert_type'])
    essay_train_encoded = data_train_pd['essay'].apply(lambda x: tokenize_pd(x, tokenizer))
    data_train_pd_encoded = data_train_pd.join(essay_train_encoded)   
    essay_dev_encoded = data_dev_pd['essay'].apply(lambda x: tokenize_pd(x, tokenizer))
    data_dev_pd_encoded = data_dev_pd.join(essay_dev_encoded)

    # normalize / scale data with minmaxscaler from [1,7] to [0,1] 
    scaler = MinMaxScaler()
    data_train_pd_encoded[['empathy', 'distress']] = scaler.fit_transform(data_train_pd_encoded[['empathy', 'distress']])
    data_dev_pd_encoded[['empathy', 'distress']] = scaler.transform(data_dev_pd_encoded[['empathy', 'distress']])

    # create dataset for each task
    # empathy
    emp_data_train_pd_encoded = data_train_pd_encoded.drop(['distress'], axis=1, inplace=False)
    emp_data_dev_pd_encoded = data_dev_pd_encoded.drop(['distress'], axis=1, inplace=False)
    # distress 
    dis_data_train_pd_encoded = data_train_pd_encoded.drop(['empathy'], axis=1, inplace=False)
    dis_data_dev_pd_encoded = data_dev_pd_encoded.drop(['empathy'], axis=1, inplace=False)


    # store which label is from what task, so that we can hand the different data points for teh tasks later on
    emp_data_train_pd_encoded['task'] = Task(name='empathy', num_labels=1, id=0, output_type='regression')
    emp_data_dev_pd_encoded['task'] = Task(name='empathy', num_labels=1, id=0, output_type='regression')
    dis_data_train_pd_encoded['task'] = Task(name='distress', num_labels=1, id=1, output_type='regression')
    dis_data_dev_pd_encoded['task'] = Task(name='distress', num_labels=1, id=1, output_type='regression')

    # rename individual target name to label
    emp_data_train_pd_encoded.rename(columns={'empathy':'label'}, inplace=True)
    emp_data_dev_pd_encoded.rename(columns={'empathy':'label'}, inplace=True)
    dis_data_train_pd_encoded.rename(columns={'distress':'label'}, inplace=True)
    dis_data_dev_pd_encoded.rename(columns={'distress':'label'}, inplace=True)

    # shuffle
    np.random.seed(params['seed'])
    emp_data_train_pd_shuffled = shuffle(emp_data_train_pd_encoded)
    emp_data_dev_pd_shuffled = shuffle(emp_data_dev_pd_encoded)
    dis_data_train_pd_shuffled = shuffle(dis_data_train_pd_encoded)
    dis_data_dev_pd_shuffled = shuffle(dis_data_dev_pd_encoded)

    # --------------------------------
    #   Create Multitasking Dataset
    # --------------------------------

    # combine dataset with samples from both tasks, best doing while still pandas
    # shuffle: single batch should contain samples from multiple tasks
    # TODO: Should we include one sample two times with two labels or on sample only with one label?
    multi_data_train = shuffle(pd.concat([emp_data_train_pd_shuffled, dis_data_train_pd_shuffled], ignore_index=True))
    multi_data_dev = shuffle(pd.concat([emp_data_dev_pd_shuffled, dis_data_dev_pd_shuffled], ignore_index=True))

    # creating dataset and dataloader from huggingface
    dataloader_train = create_dataloaders_pd(multi_data_train, batch_size=params['batch_size'])
    dataloader_dev = create_dataloaders_pd(multi_data_dev, batch_size=params['batch_size'])
    print(dataloader_dev)
    # delete end
    # ---------------------------------------------------------------

    # --------------------------------
    #   Initialize Trainer and Model
    # --------------------------------


    # --------------------------------
    #       Do training and eval
    # --------------------------------



    """
    tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args)

    model = BertMultiTask(model_args.encoder_name_or_path, task_list)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(data_args.max_eval_samples)))

            eval_datasets = new_ds

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "token_classification":
                data_collator = DataCollatorForTokenClassification(
                    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
                )
            else:
                if data_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                        tokenizer, pad_to_multiple_of=8
                    )
                else:
                    data_collator = None

            trainer.data_collator = data_collator
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_datasets)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
if __name__ == "__main1__":

    model_args = ModelArguments(encoder_name_or_path="bert-base-cased")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="/tmp/test",
        learning_rate=2e-5,
        num_train_epochs=3,
        overwrite_output_dir=True,
    )
    data_args = DataTrainingArguments(max_seq_length=512)
    run(model_args, data_args, training_args)
"""

if __name__ == '__main__':
    # check if there is an input argument
    #args = sys.argv[1:]  # ignore first arg as this is the call of this python script
    #possible_empathy_types = ['empathy', 'distress']

    params = {'bert_type': "bert-base-uncased",
            'learning_rate': 2e-5,
            'epochs': 3,
            'batch_size': 4,
            'do_train': True,
            'do_eval': True,
            'seed': 17,
            'root_dir': "",
            'output_dir': "/output",
            }
    
    print(f'Using this parameters:\n {params}')

    run(params)

