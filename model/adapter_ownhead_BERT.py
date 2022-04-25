# ---------- Sources ----------
#
# [1] Tokenizing and usage of BERT: 
#   https://huggingface.co/docs/transformers/training
# [2] Bert for regression task: 
#   https://medium.com/@anthony.galtier/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
# [3] Adapter versions config:
#   https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
# [4] Unify parameter efficient training
#   https://github.com/jxhe/unify-parameter-efficient-tuning
#
# ------------------------------

# utils
from logging import root
import copy
from transformers.adapters import AutoAdapterModel, RobertaAdapterModel
from transformers.adapters import MAMConfig, AdapterConfig, PrefixTuningConfig, ParallelConfig, HoulsbyConfig
from transformers.adapters import configuration as adapter_configs
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

# TODO: I need to structure for adapters
# TODO: Use Trainer / Adaptertrainer
class RegressionModelAdapters(nn.Module):
    def __init__(self, settings): #bert_type, task_type, adapter_config, activation_func='relu', dropout=0.5):
        super(RegressionModelAdapters, self).__init__()
        D_in, D_out = 768, 1 
        self.bert_type = settings['bert_type']
        self.adapter_name = settings['empathy_type'] + '_adapter'
        self.adapter_config = get_adapter_config(settings['adapter_type'])


        self.bert = RobertaAdapterModel.from_pretrained(self.bert_type)

        # Enable adapter training
        # task adapter - only add if not existing
        if self.adapter_name not in self.bert.config.adapters:
            print('adding adapter')
            self.bert.add_adapter(self.adapter_name, config=self.adapter_config, set_active=True)
        #self.bert.set_active_adapters(adapter_name)
        self.bert.train_adapter(self.adapter_name)  # set adapter into training mode and freeze parameters in the transformer model

        # print frozen parameters
        if False:
            names = [n for n, p in self.bert.named_parameters()]
            paramsis = [param for param in self.bert.parameters()]
            for n, p in zip(names, paramsis):
                print(f"{n}: {p.requires_grad}")
        
        self.regression_head = model_utils.RegressionHead(D_in=D_in, D_out=D_out, activation_func=settings['activation'], dropout=settings['dropout'])

        self.bert_parameter_count = model_utils.count_updated_parameters(self.bert.parameters())
        self.head_parameter_count = model_utils.count_updated_parameters(self.regression_head.parameters())

    def forward(self, input_ids, attention_masks):
        bert_outputs = self.bert(input_ids, attention_masks)
        outputs = self.regression_head(bert_outputs)
        return outputs

    def reset_head_weights(self):
        for module in self.regression_head.children():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
        
  

def get_adapter_config(config_name, print_config=True):
    """available adapters from adapter hub (18.04.22):
    (They can be used passing a string)
    
    ADAPTER_CONFIG_MAP = {
    "pfeiffer": PfeifferConfig(),  # Pfeiffer2020
    "houlsby": HoulsbyConfig(),
    "pfeiffer+inv": PfeifferInvConfig(),
    "houlsby+inv": HoulsbyInvConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "compacter": CompacterConfig(),
    "prefix_tuning": PrefixTuningConfig(),  # Li and Liang (2021)
    "prefix_tuning_flat": PrefixTuningConfig(flat=True),
    "parallel": ParallelConfig(),  # He2021
    "scaled_parallel": ParallelConfig(scaling="learned"),
    "mam": MAMConfig(),  # He2021
    }
    
    mam config is the same as according to [3]
    config = ConfigUnion(
        PrefixTuningConfig(bottleneck_size=800),
        ParallelConfig(),
        )
    """

    # load the predefined adapter configurations from the hub
    configs_dict = copy.deepcopy(adapter_configs.ADAPTER_CONFIG_MAP)

    # create own config options using some configs from here: https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
    myprefixtuning_config = PrefixTuningConfig(flat=False, prefix_length=30, bottleneck_size=200)
    configs_dict['myprefixtuning'] = myprefixtuning_config

    # adapted from from He2021; with reduction factor = 1.5 it comes closer to 512 dim than with reduction factor of 2 (from 768 to 384)
    # however, it cant be an integer, so we need to leave it like that
    mymam_config = MAMConfig(PrefixTuningConfig(bottleneck_size=30), ParallelConfig(reduction_factor=2))
    configs_dict['mymam'] = mymam_config
    
    # select config
    if config_name in configs_dict.keys():
        config = configs_dict[config_name]
    else:
        print(f'\nMyWarning: Could not find an adapter configuration for {config_name}. Please select one of the following:\n {configs_dict.keys()}\n')
        sys.exit(-1)
    if print_config: print(config)
    """
    Source: Flexible configurations with ConfigUnion 
    https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/
    
    from transformers.adapters import AdapterConfig, ConfigUnion

    config = ConfigUnion(
        AdapterConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
        AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
    )
    model.add_adapter("union_adapter", config=config)
    """
    return config


def run(settings, root_folder=""):

    # Set seed
    torch.manual_seed(settings['seed'])
    
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        device = torch.device("cpu")

    # --- init model ---
    print('------------ initializing Model ------------')

    model = RegressionModelAdapters(settings)
    model.to(device)
    print(model)
    model, history = model_utils.run_model(model, settings, device, root_folder="", model_type=RegressionModelAdapters)
    return model, history


if __name__ == '__main__':
    # check if there is an input argument
    args = sys.argv[1:]  # ignore first arg as this is the call of this python script

    settings = utils.arg_parsing_to_settings(args, learning_rate=2e-5, batch_size=16, epochs=10, save_settings=True, bert_type='roberta-base', weight_decay=0.01, dropout=0.2, use_scheduler=True)
    # ---- end function ----
    
    run(settings=settings)

