# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
#
# ------------------------------

# utils
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Transformers, torch and model utils
from transformers import BertModel, BertConfig, BertForSequenceClassification, AutoModel
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from transformers import logging



# import own module
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils

class BertRegressor(nn.Module):
    # source (changed some things): [2]  
    # https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    def __init__(self, drop_rate=0.2, bert_type="bert-base-uncased"):
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1
        self.bert = BertModel.from_pretrained(bert_type)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs


def pd_to_dataset(data_df):
    """Create hugginface dataset from pandas dataframe

   Args:
        data_df (pd.DataFrame): _description_

    Returns:
        Dataset: The huggingface dataset from datasets.Dataset
    """
    data_df = Dataset.from_pandas(data_df)
    return data_df


def pd_to_datasetdict(train, dev):
    """Create dataset dictionary with train and dev split from two pandas dataframes

    Args:
        data_df (pd.DataFrame): _description_
        data_dev (pd.DataFrame): _description_

    Returns:
        Datasetdict: The huggingface datasets dictionary from datasets.DatasetDict with train and dev
    """
    dataset_train = Dataset.from_pandas(train)
    dataset_dev = Dataset.from_pandas(dev)
    whole_dataset = DatasetDict({'train': dataset_train, 'dev': dataset_dev})
    return whole_dataset

    
def tokenize(batch, tokenizer):
    # Source: [1] - https://huggingface.co/docs/transformers/training
    return tokenizer(batch['essay'], padding='max_length', truncation=True, max_length=512)


def create_dataloaders(inputs, masks, labels, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def old_train(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2): 
    # TODO: delete
    # Source: [2]
        for epoch in range(epochs):
            model.train()
            for step, batch in enumerate(train_dataloader): 
                print(step)  
                batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
                model.zero_grad()
                outputs = model(batch_inputs, batch_masks)           
                loss = loss_function(outputs.squeeze(), 
                                batch_labels.squeeze())
                loss.backward()
                clip_grad_norm(model.parameters(), clip_value)
                optimizer.step()
                scheduler.step()
                    
        return model

def train(model, train_dataloader, dev_dataloader, epochs, optimizer, scheduler, loss_function, device, clip_value=2):
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
    for epoch_i in range(epochs):
        # -------------------
        #      Training 
        # -------------------
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_batch = time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_count = 0, 0, 0
        history = pd.DataFrame(columns=['dev_loss', 'dev_r2', 'dev_corr'])
        
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
            total_loss += loss.item()


            optimizer.zero_grad()
            loss.backward()

            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step() 

            # backward
            #optimizer.zero_grad()
            #loss.backward()

            # adam step
            # optimizer.step()
        
            # Print the loss values and time elapsed for every 1000 batches
            if True: # (step % 1 == 0) or (step == len(train_dataloader) - 1):
            
                # Calculate time elapsed
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_count:^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_count = 0, 0
                t0_batch = time.time()
            
        # -------------------
        #     Validation 
        # -------------------
        # calculate, print and store the metrics
        dev_loss, dev_r2, dev_corr = evaluate(model, loss_function, dev_dataloader, device)
        history = history.append({'dev_loss':dev_loss, 'dev_r2':dev_r2, 'dev_corr':dev_corr}, ignore_index=True)
        #print(f"Epoch: {epoch_i + 1:^7} | dev_corr: {dev_corr} | dev_loss: {dev_loss} | dev_r2: {dev_r2}")
        print(f"Epoch: {epoch_i + 1:^7} | dev_corr: {dev_corr}")
    
    return model, history


def evaluate(model, loss_function, test_dataloader, device):
    # Source: [2]
    # adapted by adding the pearson correlation
    model.eval()
    all_outputs, all_labels = np.array([]), np.array([])
    dev_loss, dev_r2, dev_corr = [], [], []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        # -- loss --
        loss = loss_function(outputs, batch_labels)
        dev_loss.append(loss.item())
        # -- r2 --
        r2 = r2_score(outputs, batch_labels)
        dev_r2.append(r2.item())
        all_outputs = np.concatenate((all_outputs, outputs.detach().cpu().numpy()), axis = None)
        all_labels = np.concatenate((all_labels, batch_labels.detach().cpu().numpy()), axis = None)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    print(all_outputs)
    print(all_labels.shape)
    print(all_outputs.flatten().shape)
    # -- pearson correlation --
    corr, _ = score_correlation(all_outputs, all_labels)
    dev_corr.append(corr)

    return dev_loss, dev_r2, dev_corr


def r2_score(outputs, labels):
    # Source: [2]
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


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


def run():
    #logging.set_verbosity_warning()
    #logging.set_verbosity_error()
    
    # -------------------
    #     parameters
    # -------------------

    bert_type = "bert-base-uncased"
    my_seed = 17
    batch_size = 16
    epochs = 2

    # -------------------
    #   load data
    # -------------------
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder="data/")
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)
    
    # --- Create hugginface datasets ---
    # TODO: Use all data later on
    data_train = pd_to_dataset(data_train_pd)
    data_dev = pd_to_dataset(data_dev_pd)

    #  Create hugginface datasetsdict
    # data_train_dev = pd_to_datasetdict(data_train, data_dev)
        
    # -------------------
    #   preprocess data
    # -------------------

    # --- tokenize data ---
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    data_train_encoded = data_train.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=None)

    # --- shuffle data ---
    data_train_encoded_shuff = data_train_encoded.shuffle(seed=my_seed)
    data_dev_encoded_shuff = data_dev_encoded.shuffle(seed=my_seed)

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
    scaler_empathy = MinMaxScaler()
    label_scaled_empathy_train = scaler_empathy.fit_transform(label_empathy_train)
    label_scaled_empathy_dev = scaler_empathy.transform(label_empathy_dev)
    # make own for distress as std counts in transformation and it might be different for distress than empathy
    scaler_distress = MinMaxScaler()
    label_scaled_distress_train = scaler_distress.fit_transform(label_distress_train)
    label_scaled_distress_dev = scaler_distress.transform(label_distress_dev)

    # --- create dataloader ---
    # for empathy
    dataloader_emp_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_empathy_train, batch_size)
    dataloader_emp_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, batch_size)
    # for distress
    dataloader_dis_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_distress_train, batch_size)
    dataloader_dis_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, batch_size)

    # -------------------
    #  initialize model 
    # -------------------
    # source for creating and training model: [2] 
    #   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    


    # --- init model ---
    model = BertRegressor(drop_rate=0.2, bert_type=bert_type)

    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    # --- optimizer ---
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    # scheduler
    total_steps = len(dataloader_emp_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)

    # epochs
    loss_function = nn.MSELoss()
   
    model, history = train(model, dataloader_emp_train, dataloader_emp_dev, epochs, optimizer, scheduler, loss_function, device, clip_value=2)
    history.to_csv('history.csv')
    
    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()

    # Initializing a model from the bert-base-uncased style configuration
    model = BertModel(configuration)

    # Accessing the model configuration
    #configuration = model.config
    print('Done')


if __name__ == '__main__':
    run()

