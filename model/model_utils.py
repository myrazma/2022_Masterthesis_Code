# containing utils like the regression head to share among different models
# e.g. here are methods / classes that should stay the same or can be used among different models

import torch.nn as nn
import pandas as pd
import numpy as np
import time
import copy
import math
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from scipy.stats import pearsonr

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
        print(f'-------------- pool output size: {pool_out_size} --------------')
        first_hid = int(np.ceil(D_in / 2))  # 384
        self.bert_head = nn.Sequential(
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.Linear(pool_out_size, 128),
            activation_layer,
            nn.Dropout(0.5))

        self.regressor = nn.Sequential(
            nn.Linear(128, 10),
            activation_layer,
            nn.Dropout(0.5),
            nn.Linear(10, 1))


    def forward(self, bert_outputs):
        bert_output = bert_outputs[1]
        bert_head_output = self.bert_head(bert_output)
        outputs = self.regressor(bert_head_output)
        return outputs


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


def train_model(model, train_dataloader, dev_dataloader, epochs, optimizer, scheduler, loss_function, device, clip_value=2, early_stop_toleance=2, use_early_stopping=True, use_scheduler=False):
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

            # backward
            #optimizer.zero_grad()
            #loss.backward()

            # adam step
            # optimizer.step()
        
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