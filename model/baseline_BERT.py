# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
#
# ------------------------------

# Transformers, torch and model utils
from transformers import BertModel, RobertaModel
import torch
import torch.nn as nn



# import own module
import model_utils

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils
import preprocessing

class BertRegressor(nn.Module):
    # source (changed some things): [2]  
    # https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
    
    def __init__(self, bert_type="bert-base-uncased", train_only_bias=False, train_bias_mlp=False, activation_func='relu', dropout=0.5):
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1
        self.bert_type = bert_type
        self.train_only_bias = train_only_bias

        self.__init_bert()
        self.regression_head = model_utils.RegressionHead(D_in=D_in, D_out=D_out, activation_func=activation_func, dropout=dropout)

        # get the size of the model parameters (head and bert separated)
        self.bert_parameter_count = model_utils.count_updated_parameters(self.bert.parameters())
        self.head_parameter_count = model_utils.count_updated_parameters(self.regression_head.parameters())

    def forward(self, input_ids, attention_masks):
        bert_outputs = self.bert(input_ids, attention_masks)
        outputs = self.regression_head(bert_outputs)
        return outputs

    def __init_bert(self):
        if self.bert_type == 'roberta-base':
            self.bert = RobertaModel.from_pretrained(self.bert_type)
        else:
            self.bert = BertModel.from_pretrained(self.bert_type)

        if self.train_only_bias == 'all' or self.train_only_bias == 'mlp':
            print(f'\n------------ Train only the bias: {self.train_only_bias} ------------\n')
            bias_filter = lambda x: 'bias' in x
            if self.train_only_bias == 'mlp':  # train only the mlp layer (excluding all biases in the attention layers)
                bias_filter = lambda x: 'bias' in x and not 'attention' in x

            names = [n for n, p in self.bert.named_parameters()]
            params = [param for param in self.bert.parameters()]
            for n, p in zip(names, params):
                if bias_filter(n):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def reset_model_weights(self):
        self.__init_bert()  # reset bert to pre trained state

        #for layer in self.bert.children():
        #    layer.parameter
        #    break
        for layer in self.regression_head.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
        

def run(settings, root_folder=""):

    # Set seed
    torch.manual_seed(settings['seed'])
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    # -------------------
    #  initialize model 
    # -------------------
    # --- init model ---
    print('------------ initializing Model ------------')
    model = BertRegressor(bert_type=settings['bert_type'], train_only_bias=settings['train_only_bias'], activation_func=settings['activation'], dropout=settings['dropout'])
    model.to(device)
    model, history = model_utils.run_model(model, settings, device, root_folder="")
    return model, history


if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script

    settings = utils.arg_parsing_to_settings(args, early_stopping=False, learning_rate=2e-5, batch_size=16, bert_type='roberta-base', epochs=10, weight_decay=0.01, save_settings=True, use_scheduler=True, dropout=0.2, kfold=0)
    # ---- end function ----
    
    run(settings=settings)

