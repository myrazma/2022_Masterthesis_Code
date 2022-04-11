# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
#
# ------------------------------

# utils
from logging import root
import time
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
# Transformers, torch and model utils
from transformers import AutoTokenizer, BertModel, BertConfig, BertForSequenceClassification, AutoModel, RobertaModel
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AdamW
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import get_linear_schedule_with_warmup
import transformers.adapters as adapters
from transformers.adapters import AutoAdapterModel, RobertaAdapterModel, PredictionHead, MAMConfig, AdapterConfig
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
from transformers import logging



# import own module
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils

# TODO: I need to structure for adapters
# TODO: Use Trainer / Adaptertrainer
class RegressionModelAdapters(nn.Module):
    def __init__(self, bert_type, task_type):
        super(RegressionModelAdapters, self).__init__()
        D_in = 768
        Bert_out = 100
        Multi_in = Bert_out + 1
        Hidden_Regressor = 50
        D_out = 1

        head_name = task_type + '_head'
        adapter_name = task_type + '_adapter'
        self.bert = RobertaAdapterModel.from_pretrained(bert_type)

            
        # Enable adapter training
        self.bert_head = mySequential(
            nn.Dropout(0.2),
            nn.Linear(D_in, D_out))

        # TODO
        #regr_head = RegressionHead(self.bert, head_name)  # self.bert_head
        #self.bert.register_custom_head(head_name, regr_head)
        #self.bert.add_custom_head(head_type=head_name, head_name='_' + head_name)
        #self.bert.add_classification_head(adapter_name, num_labels=1)  # if label is one, regression is used
        
        # task adapter - only add if not existing
        if adapter_name not in self.bert.config.adapters:
            self.bert.add_adapter(adapter_name, config="pfeiffer", set_active=True)
        #self.bert.set_active_adapters(adapter_name)
        self.bert.train_adapter(adapter_name)  # set adapter into training mode and freeze parameters in the transformer model
        #self.bert.active_adapters = adapter_name
        #self.bert = RobertaAdapterModel.from_pretrained(bert_type)
        #self.bert.add_adapter("mam_adapter", config=mam_config)
        #self.bert.register_custom_head("regression_head", CustomHead)
        #self.bert.add_custom_head(head_type="regression_head", head_name="regression_head_1", **kwargs)
        #self.bert.add_adapter("regression_head_1", config="pfeiffer")
        #self.bert.set_active_adapters("regression_head_1"
        
        # print frozen parameters
        if False:
            names = [n for n, p in self.bert.named_parameters()]
            paramsis = [param for param in self.bert.parameters()]
            for n, p in zip(names, paramsis):
                print(f"{n}: {p.requires_grad}")
            
        self.bert_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(D_in, D_out))

        self.regressor = nn.Sequential(
            nn.Linear(Multi_in, Hidden_Regressor),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(Hidden_Regressor, 10),
            nn.Linear(10, D_out))

    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        # TODO I need to hand lexical features somwhere dont I?
        bert_output = outputs[1]
    
        after_bert_outputs = self.bert_head(bert_output)
    
        # concat bert output with multi iput - lexical data
        # combine bert output (after short ffn) with lexical features
        #concat = torch.cat((after_bert_outputs, lexical_features), 1)
        #outputs = self.regressor(concat)
        return after_bert_outputs

class mySequential(nn.Sequential):
    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        input = outputs
        for module in self._modules.values():
            print((outputs[0],) + outputs[2:])
            #print(outputs)
            print(cls_output)
            print(module)
            input = module(*input)
        return input


class MyPredictionHead(nn.Sequential):
    def __init__(self, name):
        super().__init__()
        self.config = {}
        self.name = name

    def build(self, model):
        model_config = model.config

        dropout_prob = self.config.get("dropout_prob", model_config.hidden_dropout_prob)
        #for i, module in enumerate(pred_head):
        self.add_module(0, nn.Dropout(dropout_prob))
        self.add_module(1, nn.Linear(model_config.hidden_size, model_config.hidden_size))
        self.add_module(2, nn.Linear(model_config.hidden_size, 1))

        self.apply(model._init_weights)
        self.train(model.training)  # make sure training mode is consistent

    def get_output_embeddings(self):
        return None  # override for heads with output embeddings


    
class RegressionHead(MyPredictionHead):
    # source https://docs.adapterhub.ml/prediction_heads.html#adaptermodel-classes
    def __init__(
        self,
        model,
        head_name,
        **kwargs,
    ):
        super().__init__(head_name)
        # TODO: Add multli input, if wanted
        D_in = 768
        Bert_out = 100
        Multi_in = Bert_out +1
        Hidden_Regressor = 50
        D_out = 1

        self.bert_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(D_in, D_out))
            #nn.Linear(D_in, Bert_out))

        self.regressor = nn.Sequential(
            nn.Linear(Multi_in, Hidden_Regressor),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(Hidden_Regressor, 10),
            nn.Linear(10, D_out))
        
        #self.build(model)
    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        if cls_output is None:
            if self.config["use_pooler"]:
                cls_output = kwargs.pop("pooled_output")
            else:
                cls_output = outputs[0][:, 0]
        logits = super().forward(cls_output)
    """
    def forward(self, head_inputs, head_name, attention_mask=None,return_dict=None,pooled_output=None,**kwargs,):

        #cls_output = kwargs.pop("pooled_output")
        #print(type(outputs))
        #if cls_output is None:
        #cls_output = outputs[0][:, 0]
        print(head_name)
        print(head_inputs[0][:, 0])
        logits = super().forward(head_inputs[0][:, 0])
        loss = None
        labels = kwargs.pop("labels", None)
        print(labels)
        print('-----------------------')
        #print(type(attention_mask))
        #outputs = self.bert(input_ids, attention_masks)
        bert_output = head_inputs[1]

        # concat bert output with multi iput - lexical data
        after_bert_outputs = self.bert_head(bert_output)
    
        # combine bert output (after short ffn) with lexical features
        #concat = torch.cat((after_bert_outputs, lexical_features), 1)
        #outputs = self.regressor(concat)
        return after_bert_outputs"""


class MyDataset(PyTorchDataset):
    def __init__(self, input_ids, attention_mask, labels, device):
        """Initializer for SeqDataset
        Args:
            seq (np.array) [x, y]: The sequences of shape (sample_size, max_seq_size)
            labels (np.array) [x, y]: The labels of shape (sample_size, max_seq_size)
        """
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor).to(device)
        self.input_ids = torch.from_numpy(input_ids).type(torch.FloatTensor).to(device)
        self.attention_mask = torch.from_numpy(attention_mask).type(torch.LongTensor).to(device)

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

    
def tokenize(batch, tokenizer, column):
    # Source: [1] - https://huggingface.co/docs/transformers/training
    # longest is around 200
    return tokenizer(batch[column], padding='max_length', truncation=True, max_length=256)


def create_dataloaders(inputs, masks, labels, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_dataloaders_multi_in(inputs, masks, labels, lexical_features, batch_size):
    # Source: [2]
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    lexical_features_tensor = torch.tensor(lexical_features)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, lexical_features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(model, train_dataloader, dev_dataloader, epochs, optimizer, scheduler, loss_function, device, clip_value=2, bert_update_epochs=10, early_stop_toleance=2):
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

    history = pd.DataFrame(columns=['epoch', 'avrg_dev_loss', 'avrg_dev_r2', 'dev_corr', 'train_corr'])

    # variables for early stopping
    # counter for loss, counter goes up, if the loss is worse (bigger) than before, will be set to 0 if the loss is smaller
    worse_loss, prev_loss, curr_loss = 0, None, 0
    model_best = None
    epoch_model_saved = 0

    for epoch_i in range(epochs):
        print('------ epoch ' + str(epoch_i) + ' ------')
        print('Beginning of epoch ' + str(int(epoch_i)) + ', worse_loss ' + str(worse_loss))

        # only train bert for 2 epochs, otherwise bert might 'forget too much'
        #if epoch_i == bert_update_epochs:
        #    for p in model.bert.parameters():
        #        p.requires_grad = False
        #    for p in model.bert.embeddings.parameters():
        #        p.requires_grad = False
        #    print('------- Bert parameter is not being updated anymore -------')

        # -------------------
        #      Training 
        # -------------------
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_batch = time.time()

        # Reset tracking variables at the beginning of each epoch
        batch_loss, batch_count = 0, 0
        total_epoch_loss = []
        
        model.train()
    
        # For each batch of training data...
        # for batch_idx, (data, labels) in enumerate(train_loader):
        
        for step, batch in enumerate(train_dataloader):  
            batch_count +=1
            #batch_inputs, batch_masks, batch_labels, batch_lexical = tuple(b.to(device) for b in batch)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            #model.zero_grad()
            scores = model(batch_inputs, batch_masks)
            print(scores)
            #loss, logits, hidden_states, attentions = scores
            print(batch_labels)
            break
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
            scheduler.step() 

            # backward
            #optimizer.zero_grad()
            #loss.backward()

            # adam step
            # optimizer.step()
        
            # Print the loss values and time elapsed for every 1000 batches
            if (step % 50 == 0) or (step == len(train_dataloader) - 1):
            
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
        model.eval()
        # calculate, print and store the metrics
        dev_loss, dev_corr = evaluate(model, loss_function, dev_dataloader, device)
        train_loss, train_corr = evaluate(model, loss_function, train_dataloader, device)
        avrg_dev_loss, avrg_dev_r2, avrg_train_loss = None, None, None
        if len(dev_loss) > 0:
            avrg_dev_loss = sum(dev_loss)/len(dev_loss)
        if len(total_epoch_loss) > 0:
            # remove nans or infs for calculation
            filtered_train_loss = [val for val in total_epoch_loss if not (math.isinf(val) or math.isnan(val))]   
            avrg_train_loss = sum(filtered_train_loss)/len(filtered_train_loss)
            
        current_step_df = pd.DataFrame({'epoch': int(epoch_i), 'avrg_dev_loss':avrg_dev_loss, 'dev_corr': dev_corr, 'train_corr': train_corr, 'avrg_train_loss': avrg_train_loss}, index=[0])
        history = pd.concat([history, current_step_df], ignore_index=True)

        print(f"Epoch: {epoch_i + 1:^7} | dev_corr: {dev_corr} | train_corr: {train_corr} | avrg_dev_loss: {avrg_dev_loss} | avrg_train_loss: {avrg_train_loss}")
        
        # -------------------
        #   Early stopping 
        # -------------------

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

        print('worse_loss', worse_loss)

        if int(worse_loss) == int(early_stop_toleance):
            print('early stopping at epoch', int(epoch_i))
            break
        

    # save at which state we saved the model
    model_saved_at_arr = np.zeros(history.shape[0])
    model_saved_at_arr[epoch_model_saved] = 1  # set to one at this epoch
    history['model_saved'] = model_saved_at_arr

    return model_best, history


def evaluate(model, loss_function, test_dataloader, device):
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


def run(root_folder="", empathy_type='empathy'):

    data_root_folder = root_folder + 'data/'
    output_root_folder = root_folder + 'output/'

    #logging.set_verbosity_warning()
    #logging.set_verbosity_error()
    use_gpu = False
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
        use_gpu = True
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")
        use_gpu = False
    # -------------------
    #     parameters
    # -------------------

    bert_type = "roberta-base"  # "bert-base-uncased"
    my_seed = 17
    batch_size = 4
    epochs = 2
    learning_rate = 2e-5  # 2e-5

    # -------------------
    #   load data
    # -------------------
    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)

    data_train_pd = utils.clean_raw_data(data_train_pd[:20])
    data_dev_pd = utils.clean_raw_data(data_dev_pd[:10])

    # save raw essay (will not be tokenized by BERT)
    data_train_pd['essay_raw'] = data_train_pd['essay']
    data_dev_pd['essay_raw'] = data_dev_pd['essay']
    
    # tokenize them already and create column essay_raw_tok
    data_train_pd = utils.tokenize_data(data_train_pd, 'essay_raw')
    data_dev_pd = utils.tokenize_data(data_dev_pd, 'essay_raw')
    
    # create lexical features
    fc = utils.FeatureCreator(data_root_folder=data_root_folder)
    data_train_pd = fc.create_lexical_feature(data_train_pd, 'essay_raw_tok')
    data_dev_pd = fc.create_lexical_feature(data_dev_pd, 'essay_raw_tok')

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
    tokenizer = RobertaTokenizer.from_pretrained(bert_type)

    data_train_encoded = data_train.map(lambda x: tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)
    data_dev_encoded = data_dev.map(lambda x: tokenize(x, tokenizer, 'essay'), batched=True, batch_size=None)


    # --- shuffle data ---
    data_train_encoded_shuff = data_train_encoded.shuffle(seed=my_seed)
    data_dev_encoded_shuff = data_dev_encoded.shuffle(seed=my_seed)

    # get input_ids, attention_mask and labels
    # train
    input_ids_train = np.array(data_train_encoded_shuff["input_ids"]).astype(int)
    attention_mask_train = np.array(data_train_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_train = np.array(data_train_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_train = np.array(data_train_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    lexical_emp_train = np.array(data_train_encoded_shuff["empathy_word_rating"]).astype(np.float32).reshape(-1, 1)
    lexical_dis_train = np.array(data_train_encoded_shuff["distress_word_rating"]).astype(np.float32).reshape(-1, 1)
    
    # dev
    input_ids_dev = np.array(data_dev_encoded_shuff["input_ids"]).astype(int)
    attention_mask_dev = np.array(data_dev_encoded_shuff["attention_mask"]).astype(int)
    label_empathy_dev = np.array(data_dev_encoded_shuff["empathy"]).astype(np.float32).reshape(-1, 1)
    label_distress_dev = np.array(data_dev_encoded_shuff["distress"]).astype(np.float32).reshape(-1, 1)
    lexical_emp_dev = np.array(data_dev_encoded_shuff["empathy_word_rating"]).astype(np.float32).reshape(-1, 1)
    lexical_dis_dev = np.array(data_dev_encoded_shuff["distress_word_rating"]).astype(np.float32).reshape(-1, 1)

    # --- scale labels: map empathy and distress labels from [1,7] to [0,1] ---
    label_scaled_empathy_train = utils.normalize_scores(label_empathy_train, (1,7))
    label_scaled_empathy_dev = utils.normalize_scores(label_empathy_dev, (1,7))
    label_scaled_distress_train = utils.normalize_scores(label_distress_train, (1,7))
    label_scaled_distress_dev = utils.normalize_scores(label_distress_dev, (1,7))
    
    # -------------------
    #  initialize pre trained model an 
    # -------------------
    # source for creating and training model: [2] 
    #   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    # --- load or pre-train bert here ---
    #pre_trained_emp_bert = BertRegressor()
    #pre_trained_emp_bert = torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330', map_location=torch.device('cpu'))
    #if use_gpu:
    #    pre_trained_emp_bert.load_state_dict(torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330'))
    #else:
    #    pre_trained_emp_bert.load_state_dict(torch.load(root_folder + 'output/model_' + empathy_type + '_22-03-25_1330',map_location=torch.device('cpu')))
    # get output from pre-trained bert
    #bert_outputs_emp_train = pre_trained_emp_bert(torch.tensor(input_ids_train), torch.tensor(attention_mask_train))
    #bert_outputs_emp_dev = pre_trained_emp_bert(torch.tensor(input_ids_dev), torch.tensor(attention_mask_dev))

    #emp_dev_corr, _ = score_correlation(np.array(bert_outputs_emp_dev), np.array(label_empathy_dev))
    #print(emp_dev_corr)
    #return
    # -- pearson correlation --


    # --- create dataloader ---
    # with lexical data
    # for empathy
    #dataloader_emp_train = create_dataloaders_multi_in(input_ids_train, attention_mask_train, label_scaled_empathy_train, lexical_emp_train, batch_size)
    #dataloader_emp_dev = create_dataloaders_multi_in(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, lexical_emp_dev, batch_size)
    # for distress
    #dataloader_dis_train = create_dataloaders_multi_in(input_ids_train, attention_mask_train, label_scaled_distress_train, lexical_dis_train, batch_size)
    #dataloader_dis_dev = create_dataloaders_multi_in(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, lexical_dis_dev, batch_size)

    # without lexical data
    # for empathy
    dataloader_emp_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_empathy_train, batch_size)
    dataloader_emp_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, batch_size)
    # for distress
    dataloader_dis_train = create_dataloaders(input_ids_train, attention_mask_train, label_scaled_distress_train, batch_size)
    dataloader_dis_dev = create_dataloaders(input_ids_dev, attention_mask_dev, label_scaled_distress_dev, batch_size)

    # TODO create for distress and select accoridng to method input from run()
    pytorch_dataset_emp_train = MyDataset(input_ids_train, attention_mask_train, label_scaled_empathy_train, device)
    pytorch_dataset_emp_dev = MyDataset(input_ids_dev, attention_mask_dev, label_scaled_empathy_dev, device)
    

    # --- choose dataset ---
    # per default use empathy label
    dataloader_train = dataloader_emp_train
    dataloader_dev = dataloader_emp_dev
    display_text = 'Using empathy data'
    if empathy_type == 'distress':
        dataloader_train = dataloader_dis_train
        dataloader_dev = dataloader_dis_dev
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')


    # --- init model ---
    print('------------ initializing Model ------------')
    # ------------------------ TODO: added 
    #model = RobertaAdapterModel.from_pretrained(bert_type)
    #model.register_custom_head("regression_head", CustomHead)
    #model.add_custom_head(head_type="regression_head", head_name="regression_head_1")
    #model.add_adapter("regression_head_1", config="pfeiffer")
    #model.set_active_adapters("regression_head_1")
    #model.train_adapter("regression_head_1")
    """
    #model = RegressionModelAdapters(bert_type=bert_type, task_type=empathy_type)
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=1,
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )
    #model = RobertaAdapterModel.from_pretrained(bert_type)
    # TODO
    head_name = empathy_type + '_head'
    adapter_name = empathy_type + '_adapter'
    regr_head = RegressionHead(model, head_name)
    model.register_custom_head(head_name, regr_head)
    model.add_custom_head(head_type=head_name, head_name='_' + head_name)
    #model.add_classification_head(adapter_name, num_labels=1)  # if label is one, regression is used
    model.add_adapter(adapter_name, config="pfeiffer")
    model.set_active_adapters(adapter_name)
    model.train_adapter(adapter_name)  # set adapter into training mode and freeze parameters in the transformer model

    # -------------------------- added done --------------------------
    #model = BertMultiInput(drop_rate=0.2, bert_type=bert_type)
    model.to(device)

    # -------------------------------
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=200,
        output_dir="./outptu",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )
    
    def score_correlation(y_pred, y_true):
        r, p = pearsonr(y_true, y_pred)
        return r, p

    def compute_correlation(p: EvalPrediction):
        print(p)
        r, p = pearsonr(p.label_ids, p.predictions)
        return {"corr": r[0]}

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"corr": (preds == p.label_ids).mean()}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_emp_train,
        eval_dataset=pytorch_dataset_emp_dev,
        compute_metrics=compute_correlation,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    return
    """

    #model = BertMultiInput(drop_rate=0.2, bert_type=bert_type)
    model = RegressionModelAdapters(bert_type=bert_type,task_type=empathy_type)
    model.to(device)
    # -------------------------------
    # --- optimizer ---
    # low learning rate to not get into catastrophic forgetting - Sun 2019
    # default epsilon by pytorch is 1e-8
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    # scheduler
    total_steps = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=total_steps)

    # epochs
    loss_function = nn.MSELoss()
   
    model, history = train(model, dataloader_train, dataloader_dev, epochs, optimizer, scheduler, loss_function, device, clip_value=2)
    history.to_csv(output_root_folder + 'history_multiinput_' + empathy_type + '.csv')
    
    torch.save(model.state_dict(), output_root_folder + 'model_multiinput_' + empathy_type)
    print('Done')
    return model, history


if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script
    possible_empathy_types = ['empathy', 'distress']
    if len(args) > 0:
        empathy_type = args[0]
        if empathy_type not in possible_empathy_types:
            print(f"The possible empathy types are: {possible_empathy_types}. Your arg was: {empathy_type}. Exiting.")
            sys.exit(-1)
    else:
        empathy_type = 'empathy'
    
    print(f'\n------------ Using {empathy_type} as argument. ------------\n')
    
    run(empathy_type=empathy_type)

