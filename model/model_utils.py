# containing utils like the regression head to share among different models
# e.g. here are methods / classes that should stay the same or can be used among different models

import torch.nn as nn
import pandas as pd
import numpy as np
import time
import copy
import math
import torch
from transformers import AutoTokenizer, BertModel, BertConfig, BertForSequenceClassification, AutoModel, RobertaModel
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, RobertaTokenizer
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import get_linear_schedule_with_warmup
import transformers.adapters as adapters
from transformers.adapters import AutoAdapterModel, RobertaAdapterModel, PredictionHead
from transformers.adapters import MAMConfig, AdapterConfig, PrefixTuningConfig, ParallelConfig, HoulsbyConfig
from transformers.adapters import configuration as adapter_configs
from sklearn.model_selection import KFold
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, random_split,SubsetRandomSampler, ConcatDataset
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset as PyTorchDataset
from torch.optim import AdamW

from scipy.stats import pearsonr

# import own module
from baseline_BERT import BertRegressor
from adapter_ownhead_BERT import RegressionModelAdapters
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils
import preprocessing



class RegressionHead(nn.Module):
    """Regression head for Bert model

    Args:
        nn (nn.Module): Inherit from nn.Module
    """

    def __init__(self, dropout=0.2, D_in=768, D_hidden1=100, D_hidden2=10, D_out=1, activation_func='relu'):
        super(RegressionHead, self).__init__()


        if activation_func == 'tanh':
            print('Using Tanh as activation function.')
            activation_layer = nn.Tanh()
        else:
            print('Using ReLU as activation function.')
            activation_layer = nn.ReLU()  # per default_use relu
            
            # calcuate output size of pooling layer
        padding = 0
        dilation = 1
        stride = 2
        kernel_size = 3
        pool_out_size = int(np.floor((D_in + 2 * padding - dilation * (kernel_size-1)-1)/stride +1))
        #print(f'-------------- pool output size: {pool_out_size} --------------')
        first_hid = int(np.ceil(D_in / 2))  # 384
        self.bert_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 100))

        self.regressor = nn.Sequential(
            nn.Linear(100, 10),
            activation_layer,
            nn.Linear(10, 1))

        #self.bert_head = nn.Sequential(
        #    nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        #    nn.Linear(pool_out_size, 128),
        #    activation_layer,
        #    nn.Dropout(0.5))

        #self.regressor = nn.Sequential(
        #    nn.Linear(128, 10),
        #    activation_layer,
        #    nn.Dropout(0.5),
        #    nn.Linear(10, 1))


    def forward(self, bert_outputs):
        bert_output = bert_outputs[1]
        bert_head_output = self.bert_head(bert_output)
        outputs = self.regressor(bert_head_output)
        return outputs


class MyDataset(PyTorchDataset):
    def __init__(self, input_ids, attention_mask, labels, device):
        """Initializer for SeqDataset
        Args:
            seq (np.array) [x, y]: The sequences of shape (sample_size, max_seq_size)
            labels (np.array) [x, y]: The labels of shape (sample_size, max_seq_size)
        """
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor)
        self.input_ids = torch.from_numpy(input_ids).type(torch.FloatTensor)
        self.attention_mask = torch.from_numpy(attention_mask).type(torch.LongTensor)

    def __len__(self):
        """Implement len function of type Dataset
        Returns:
            int: The length of the dataset
        """
        return len(self.labels)
            
    def __getitem__(self, idx):
        """Implement get_item function of type Dataset
        Args:
            idx (int): The index of the item to get
        Returns:
            tensor [y], tensor [y]: The sequence, The labels
        """
        item = {}
        item['attention_masks'] = self.attention_mask[idx].int()
        item['input_ids'] = self.input_ids[idx].int()
        item['label'] = self.labels[idx].float()
        return item


def count_updated_parameters(model_params):
    """Count the parameters of the model that are updated (requires_grad = True)

    Args:
        model_params (_type_): The model parameters

    Returns:
        int: The number of parameters updated
    """
    model_size = 0
    for p in model_params:
        if p.requires_grad:  # only count if the parameter is updated during training
            model_size += p.flatten().size()[0]
    return model_size


def train_model(model, train_dataloader, dev_dataloader, epochs, optimizer, scheduler, loss_function, device, clip_value=2, early_stop_toleance=2, use_early_stopping=False, use_scheduler=False):
    """Train the model on train dataset and evelautate on the dev dataset
    Source parly from [2]
    Args:
        model (BILSTM): The model that should be trained
        train_loader (DataLoader): The DataLoader of the train SeqDataset
        dev_loader (DataLoader): The DataLoader of the dev SeqDataset
        epochs (int): The number of epochs
        optimizer (Optimizer): The optimizer object
        criterion (nn.CrossEntropyLoss): The Loss function, here cross-entropy loss
        pad_label_id (int): The label id of the padding label (based on label lookup table)
        output_dim (int): The output dimension of the model
    Returns:
        BILSTM: The trained model
    """
    print("Start training...\n")

    history = pd.DataFrame(columns=['epoch', 'avrg_dev_loss', 'dev_corr', 'train_corr'])
    # init variables for early stopping
    worse_loss, epoch_model_saved = 0, 0
    model_best = None
    for epoch_i in range(epochs):
        # -------------------
        #      Training 
        # -------------------
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_batch = time.time()
        t0_epoch = time.time()

        # Reset tracking variables at the beginning of each epoch
        batch_loss, batch_count = 0, 0
        total_epoch_loss = []
        
        model.train()
    
        # For each batch of training data...
        # for batch_idx, (data, labels) in enumerate(train_loader):
        
        for step, batch in enumerate(train_dataloader):  
            batch_count +=1
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            #model.zero_grad()
            scores = model(batch_inputs, batch_masks)
            
            # make predictions into right shape
            #predictions = scores.view(-1, scores.shape[-1])
            #tags = labels.view(-1)
            loss = loss_function(scores.squeeze(), batch_labels.squeeze())
            # loss = criterion(predictions, tags)
            batch_loss += loss.item()
            total_epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            if use_scheduler: scheduler.step() 
        
            # Print the loss values and time elapsed for every 1000 batches
            if ((step % 50 == 0) and (step != 0)) or (step == len(train_dataloader) - 1):
            
                # Calculate time elapsed
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_count:^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_count = 0, 0
                t0_batch = time.time()

        epoch_time_elapsed = time.time() - t0_epoch
            
        # -------------------
        #     Validation 
        # -------------------
        model.eval()
        # calculate, print and store the metrics
        dev_loss, dev_corr = evaluate_model(model, loss_function, dev_dataloader, device)
        train_loss, train_corr = evaluate_model(model, loss_function, train_dataloader, device)
        avrg_dev_loss, avrg_dev_r2, avrg_train_loss = None, None, None
        if len(dev_loss) > 0:
            avrg_dev_loss = sum(dev_loss)/len(dev_loss)
        if len(total_epoch_loss) > 0:
            # remove nans or infs for calculation
            filtered_train_loss = [val for val in total_epoch_loss if not (math.isinf(val) or math.isnan(val))]   
            avrg_train_loss = sum(filtered_train_loss)/len(filtered_train_loss)

        current_step_df = pd.DataFrame({'epoch': int(epoch_i), 'avrg_dev_loss':avrg_dev_loss, 'dev_corr': dev_corr, 'train_corr': train_corr, 'avrg_train_loss': avrg_train_loss, 'train_time_elapsed': epoch_time_elapsed}, index=[0])
        history = pd.concat([history, current_step_df], ignore_index=True)

        print(f"Epoch: {epoch_i + 1:^7} | dev_corr: {dev_corr} | train_corr: {train_corr} | avrg_dev_loss: {avrg_dev_loss} | avrg_train_loss: {avrg_train_loss} | training time elapsed: {epoch_time_elapsed}")
        
        # -------------------
        #   Early stopping 
        # -------------------
        if use_early_stopping:
            # was avrg_dev_loss before, but maybe correlation is also useful
            all_dev_loss = history['dev_corr'].to_numpy()
            if all_dev_loss.shape[0] > 1:  # not possible to do this in first epoch
                if all_dev_loss[-2] >= all_dev_loss[-1]:
                    worse_loss += 1
                else:
                    worse_loss = 0
            
            # save the best model according to loss
            if worse_loss > 0: # if the loss is worse than in previous training, don't save this model
                pass # do nothing
            else:
                model_best = copy.deepcopy(model)
                epoch_model_saved = int(epoch_i)

            if int(worse_loss) == int(early_stop_toleance):
                print('early stopping at epoch', int(epoch_i))
                break
        
    if model_best is None:
        # This is the case if no early stopping is used
        print('Returning last model state...')
        model_best = model
    # save at which state we saved the model
    model_saved_at_arr = np.zeros(history.shape[0])
    model_saved_at_arr[epoch_model_saved] = 1  # set to one at this epoch
    history['model_saved'] = model_saved_at_arr
    return model, history


def evaluate_model(model, loss_function, test_dataloader, device):
    # Source: [2]
    # adapted by adding the pearson correlation
    model.eval()
    all_outputs, all_labels = np.array([]), np.array([])
    dev_loss = []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        # -- loss --
        loss = loss_function(outputs, batch_labels)
        dev_loss.append(loss.item())
        # -- r2 --
        #r2 = r2_score(outputs, batch_labels)
        #dev_r2.append(r2.item())
        
        all_outputs = np.concatenate((all_outputs, outputs.detach().cpu().numpy()), axis = None)
        all_labels = np.concatenate((all_labels, batch_labels.detach().cpu().numpy()), axis = None)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    # -- pearson correlation --
    dev_corr, _ = score_correlation(all_outputs, all_labels)

    # remove inf and nan, do not count for average
    filtered_dev_loss = [val for val in dev_loss if not (math.isinf(val) or math.isnan(val))]
    return filtered_dev_loss, dev_corr


def predict(model, test_dataloader, device):
    model.eval()
    all_outputs, all_labels = np.array([]), np.array([])
    dev_loss = []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)

        all_outputs = np.concatenate((all_outputs, outputs.detach().cpu().numpy()), axis = None)
        all_labels = np.concatenate((all_labels, batch_labels.detach().cpu().numpy()), axis = None)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    # -- pearson correlation --
    test_corr, _ = score_correlation(all_outputs, all_labels)

    return all_outputs, all_labels, test_corr


def score_correlation(y_pred, y_true):
    """Correlate prediciton and true value using pearson r

    Args:
        y_pred (array): The predicted labels
        y_true (array): The true labels

    Returns:
        r, p (float, float): pearson r, p-value
    """
    r, p = pearsonr(y_true, y_pred)
    return r, p


def kfold_cross_val(model, settings, dataset_train, dataset_dev, optimizer, scheduler, loss_function, device, k=10, clip_value=2, early_stop_toleance=2, use_early_stopping=False, use_scheduler=False):
    # partly source from https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
    batch_size = settings['batch_size']
    seed = settings['seed']
    epochs = settings['epochs']
    fold_histories = []
    dataset = ConcatDataset([dataset_train, dataset_dev])

    #model = model_type(settings)
    #model.to(device)
    seg_size = int(np.ceil(len(dataset) / k))
    # create folds
    for i, fold in enumerate(range(k)):
        print(f"\n ---------------- Fold {i} ---------------- \n")
        # init model each time using model_type

        fold_range = (seg_size*i, seg_size*i + seg_size)
        if fold_range[1] >= len(dataset):  # woudl be out of bound
            fold_range = (fold_range[0], len(dataset)-1) ## replace second with the lengt of the data set - 1

        dev_indices = np.arange(fold_range[0], fold_range[1])  # get the indices of the current dev data
        train_indices = np.delete(np.arange(0, len(dataset)), dev_indices)  # the rest is the training data for this fold

        fold_dataset_train = Subset(dataset, train_indices)
        fold_dataset_dev = Subset(dataset, dev_indices)
        fold_loader_train = DataLoader(fold_dataset_train, batch_size=batch_size, shuffle=True)
        fold_loader_dev = DataLoader(fold_dataset_dev, batch_size=batch_size, shuffle=True)
     
        model, history = train_model(model, fold_loader_train, fold_loader_dev, epochs, optimizer, scheduler, loss_function, device, use_early_stopping=False, use_scheduler=settings['scheduler'])
        fold_histories.append(history)

    # average all score in the histories
    avrg_history = fold_histories[0]
    for hist in fold_histories[1:]:
        avrg_history = avrg_history + hist
    avrg_history = avrg_history / len(fold_histories)
    print('Average folds history:', avrg_history)
    return model, avrg_history  # TODO: which model to return?


def run_model(model, settings, device, model_type, root_folder=""):
    """Method for running (training, evaluation and (optional) saving) a model

    Args:
        model (nn.Module): The model (should already be all setup)
        settings (dict): The settings / parameter dictionary
        device (_type_): The device
        root_folder (str, optional): _description_. Defaults to "".

    Returns:
        nn.Module, pd.DataFrame: model, history
    """

    data_root_folder = root_folder + 'data/'
    output_root_folder = root_folder + 'output/'
    # -------------------
    #     parameters
    # -------------------

    empathy_type = settings['empathy_type']
    bert_type = settings['bert_type']
    my_seed = settings['seed']
    batch_size = settings['batch_size']
    learning_rate = settings['learning_rate']
    epochs = settings['epochs']
    train_only_bias = settings['train_only_bias']
    weight_decay = settings['weight_decay']
    use_scheduler = settings['scheduler']
    use_early_stopping = settings['early_stopping']
    activation_func = settings['activation']
    dropout = settings['dropout']

    using_roberta = False
    if bert_type == 'roberta-base':
        using_roberta = True

    # -------------------
    #   load data
    # -------------------
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)

    # --- get the tokenizer ---   
    if 'roberta' in bert_type:
        tokenizer = RobertaTokenizer.from_pretrained(bert_type)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_type)

    # --- get fully preprocessed data ---
    dataset_emp_train, dataset_emp_dev, dataset_dis_train, dataset_dis_dev = preprocessing.get_preprocessed_dataset(settings, data_train_pd, data_dev_pd, tokenizer, data_root_folder)
    # --- create dataloader ---
    # for empathy
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
    if empathy_type == 'distress':
        dataloader_train = dataloader_dis_train  # needed for k fold
        dataloader_dev = dataloader_dis_dev  # needed for k fold
        dataset_train = dataset_dis_train  # needed for k fold
        dataset_dev = dataset_dis_dev  # needed for k fold
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')

    # --- optimizer ---
    # low learning rate to not get into catastrophic forgetting - Sun 2019
    # default epsilon by pytorch is 1e-8
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)

    # scheduler
    total_steps = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)

    # epochs
    loss_function = nn.MSELoss()
    
    if settings['kfold'] > 0:  # if kfold = 0, we ar enot doing kfold
        print('\n------------ Using kfold cross validation ------------\n')
        model, history = kfold_cross_val(model, settings, dataset_train, dataset_dev, optimizer, scheduler, loss_function, device, k=settings['kfold'], use_early_stopping=False, use_scheduler=use_scheduler)
    else:
        model, history = train_model(model, dataloader_train, dataloader_dev, epochs, optimizer, scheduler, loss_function, device=device, clip_value=2, use_scheduler=use_scheduler, use_early_stopping=use_early_stopping)
    
        # add model parameter size to history
        history['bert_param_size'] = np.zeros(history.shape[0]) + model.bert_parameter_count
        history['head_param_size'] = np.zeros(history.shape[0]) + model.head_parameter_count

    print(f"\nSave settings using model name: {settings['model_name']}\n")
    history.to_csv(root_folder + 'output/history_baseline_' + empathy_type + '_' + settings['model_name'] +  '.csv')
    
    if settings['save_model']:
        print(f"\nSave model using model name: {settings['model_name']}\n")
        torch.save(model.state_dict(), root_folder + 'output/model_baseline_' + empathy_type + '_' + settings['model_name'])
    print('Done')
    return model, history