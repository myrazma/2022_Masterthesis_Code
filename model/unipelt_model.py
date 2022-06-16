""" Script for running Unipelt Model with possible feature input
Should capture:
1. Different Input:
    a. Lexicon - word average
    b. Lexicon - PCA
2. Changeable parameters for UniPELT settings (Learning rate, methods, etc.)

In here: use trainer (best from submodule/..UnifiedPELT/transformers), same like in run_emp.py


Can we maybe build a framework for this trainer to use it for other models too? So for the model of / in adapter_BERT
"""

from multiprocessing import pool
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from scipy.stats import pearsonr, spearmanr
import numpy as np

# my modules and scripts
from pathlib import Path
import sys

from torch import t
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import utils as utils
import preprocessing as preprocessing

import importlib
unipelt_transformers = importlib.import_module('submodules.2022_Masterthesis_UnifiedPELT.transformers')




"""
class MultiinputBertForSequenceClassification(BertPreTrainedModel):
    # Using the unipelt Bert implementation
    
    def __init__(self, config, feature_dim) -> None:
        super().__init__(config)
        
        # This should be UniPELT bert (but without classification head)
        self.bert = unipelt_transformers.BertModel(config)  # TODO
        # !! TODO: The input we are giving the Auto model should be in here as well!
        # TODO should be the same as BertForSequenceClassification in modeling_bert.py from the unipelt submodule
        # ecxept the added features


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # Should be the same as the classification head in the transformers library
        hidden_size = config.hidden_size + feature_dim
        self.regressor_head = nn.Linear(hidden_size, self.config.num_labels)  # TODO input dim is bert output dim + feature dim

        self.init_weights()
        

    def forward(self, input_ids, attention_masks, features):
        # TODO:
        # Lexical features should be of arbitrary length, can also be None
        # Should I set features to None in head: forward(.., features=None)


        bert_outputs = self.bert(input_ids, attention_masks)
        hidden = bert_outputs[1]

        # concat bert output with multi iput - lexical data
        #after_bert_outputs = self.after_bert(bert_output)

        if features is not None:  # if we have additiona features, concat them with our hidden features
            hidden = torch.cat((hidden, features), 1)

        outputs = self.regressor_head(hidden)

        # combine bert output (after short ffn) with lexical features
        #concat = torch.cat((after_bert_outputs, lexical_features), 1)
        #outputs = self.regressor(concat)
        #return outputs
        pass
"""
########### from unipelt transformers start #############

class MultiinputBertForSequenceClassification(unipelt_transformers.adapters.model_mixin.ModelWithHeadsAdaptersMixin, unipelt_transformers.BertPreTrainedModel):
    # Re-implement BertForSequnceClassification of UniPELT implementation but with additional feature input
    def __init__(self, config, feature_dim):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = unipelt_transformers.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        hidden_feat_size = config.hidden_size + feature_dim
        self.classifier = nn.Linear(hidden_feat_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            adapter_names=None,
            features=None,  # added by Myra Z.
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # if features are not None, concat to pooled bert output
        concat_output = torch.cat((pooled_output, features), 1) if features is not None else pooled_output  # added by Myra Z.
        logits = self.classifier(concat_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return unipelt_transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

########### from unipelt transformers end #############
"""
############ from transformers start ############
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

############ from transformers end ############
"""
def run():

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    #parser = HfArgumentParser((MyArguments, ... what else we need))

    #if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #    # If we pass only one argument to the script and it's the path to a json file,
    #    # let's parse it to get our arguments.
    #    model_args, data_args, training_args, adapter_args = parser.parse_json_file(
    #        json_file=os.path.abspath(sys.argv[1])
    #    )
    #else:
    #    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")



    # ---------------------------
    #   get the argument input
    # ---------------------------

    data_root_folder = 'data/'
    task_name = 'distress'
    use_pca_features = True
    use_lexical_features = True

    # ---------------------------
    #   Load and prepocess data
    # ---------------------------

    data_train_pd, data_dev_pd = utils.load_data(data_root_folder=data_root_folder)
    data_train_pd = utils.clean_raw_data(data_train_pd)
    data_dev_pd = utils.clean_raw_data(data_dev_pd)

    labels = data_train_pd[task_name].to_numpy().reshape(-1)

    # ---------------------------
    #       get the features
    # ---------------------------

    # The feature array will have additional features, if wanted, else it will stay None
    features = None

    fc = preprocessing.FeatureCreator(data_root_folder=data_root_folder, device=device)

    # --- create pca - empathy / distress dimension features ---
    if use_pca_features:
        emp_dim = fc.create_pca_feature(data_train_pd['essay'], task_name=task_name)
        print('emp_dim.shape', emp_dim.shape)
        emp_dim = emp_dim.reshape((-1, 1))
        #print('PEARSON R: ', pearsonr(labels, emp_dim.reshape(-1)))
        features = emp_dim if features is None else np.hstack((features, emp_dim))


    # --- create lexical features ---
    if use_lexical_features:
        data_train_pd = preprocessing.tokenize_data(data_train_pd, 'essay')
        data_dev_pd = preprocessing.tokenize_data(data_dev_pd, 'essay')
        
        fc = preprocessing.FeatureCreator(data_root_folder=data_root_folder)
        lexicon_rating = fc.create_lexical_feature(data_train_pd['essay_tok'], task_name=task_name)
        lexicon_rating = lexicon_rating.reshape((-1, 1))

        features = lexicon_rating if features is None else np.hstack((features, lexicon_rating))
        #print('PEARSON R: ', pearsonr(labels, lexicon_rating.reshape(-1)))

    feature_dim = features.shape[1] if features is not None else 0

    # ---------------------------
    #       create model
    # ---------------------------

    model = MultiinputBertForSequenceClassification(feature_dim=feature_dim)


        



if __name__ == '__main__':
    run()