import pandas as pd
import numpy as np
import sys
import json
import time
from collections import defaultdict
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')


# Load english stop words
STOPWORDS_EN = set(stopwords.words('english'))


def load_data(data_root_folder="../data/"):
    # dev and train data
    # TODO check if files are available / downloaded
    data_train = pd.read_csv(data_root_folder + "buechel_empathy/messages_train_ready_for_WS.tsv", sep='\t')
    features_dev = pd.read_csv(data_root_folder + "buechel_empathy/messages_dev_features_ready_for_WS_2022.tsv", sep='\t')
    labels_dev = pd.read_csv(data_root_folder + "buechel_empathy/goldstandard_dev_2022.tsv", sep='\t', header=None)
    
    # specify label columns
    label_columns = ['empathy', 'distress', 'emotion', 'personality_conscientiousness', 'personality_openess', 'personality_extraversion', 'personality_agreeableness', 'personality_stability', 'iri_perspective_taking',  'iri_personal_distress', 'iri_fantasy','iri_empathatic_concern']
    # since dev labels initially have no column names, add them manually
    labels_dev.columns = label_columns
    data_dev = features_dev.join(labels_dev)
    return data_train, data_dev


def clean_raw_data(data_df):
    """Preprocess raw data and dev data including the following steps:
    - remove empathy_bin and distress_bin as they are not appearing in the 
    - remove iri labels
    - remove personality labels

    Args:
        data_df (_type_): _description_
        features_dev (_type_): _description_
        labels_dev (_type_): _description_
    """
    # clean data from unnecessary files
    # remove empathy and distress bin as we dont need it here
    cols_to_drop = ['empathy_bin', 'distress_bin']
    # create loop to check if the labels are in the data or not
    for col in cols_to_drop:
        if col in data_df.columns:
            data_df.drop([col], inplace=True, axis=1)

    # remove iri and personal things a labels here
    necessary_cols = [col for col in data_df.columns if not (col.__contains__('personality') or col.__contains__('iri'))]
    # additionally remove ids for now
    necessary_cols = [col for col in necessary_cols if not col.__contains__('id')]
    data_df = data_df[necessary_cols]
    return data_df


def load_empathy_distress_lexicon(data_root_folder="../data/"):
    """Load empathy and sitress lexicon from Sedoc2019 (http://www.wwbp.org/lexica.html)
        Data should be here: data/lexicon/distress and data/lexicon/empathy

    Args:
        data_root_folder (str, optional): _description_. Defaults to "../data/".

    Returns:
        _type_: _description_
    """
    # read lexical data
    empathy_lex = pd.read_csv(data_root_folder + "lexicon/empathy/empathy_lexicon.txt")
    distress_lex = pd.read_csv(data_root_folder + "lexicon/distress/distress_lexicon.txt")

    # convert into dictionary
    empathy_lex = dict(zip(empathy_lex.word, empathy_lex.rating))
    distress_lex = dict(zip(distress_lex.word, distress_lex.rating))

    # convert into default dict (if word not in there, it will be 0)
    empathy_lex = defaultdict(int, empathy_lex)
    distress_lex = defaultdict(int, distress_lex)
    return empathy_lex, distress_lex


def load_sentenced_emotions(data_root_folder="../data/"):
    # sentencized automatic emotion tags
    sent_emotion_train = pd.read_csv(data_root_folder + "buechel_empathy/messages_train_sentencized_automatic_emotion_tags.tsv", sep='\t')
    sent_emotion_dev = pd.read_csv(data_root_folder + "buechel_empathy/messages_dev_sentencized_automatic_emotion_tags.tsv", sep='\t')
    return sent_emotion_train, sent_emotion_dev


def load_articles(data_root_folder="../data/"):
    # the news articles
    articles = pd.read_csv(data_root_folder + "buechel_empathy/articles_adobe_AMT.csv")
    return articles


def arg_parsing_to_settings(args, empathy_type = 'distress', learning_rate=2e-5, seed=17, batch_size=4, epochs=5, bert_type='roberta-base', train_only_bias='none', adapter_type="pfeiffer", model_name="", save_settings=False, early_stopping=False, weight_decay=0.01, save_model=False, use_scheduler=False, activation_func='relu', dropout=0.5, kfold=0, tensorboard=False):
    if model_name=="":  # set model name to timestamp
        model_name = time.strftime("%y-%m-%d_%H%M", time.localtime())
    # provide default settings
    settings = {'empathy_type': empathy_type,
                'learning_rate': learning_rate, 
                'seed': seed, 
                'batch_size': batch_size, 
                'epochs': epochs, 
                'bert_type': bert_type, 
                'train_only_bias': train_only_bias,
                'adapter_type': adapter_type, 
                'model_name': model_name, 
                'save_settings': save_settings, 
                'early_stopping': early_stopping, 
                'weight_decay': weight_decay, 
                'save_model': save_model, 
                'scheduler': use_scheduler,
                'activation': activation_func,
                'dropout': dropout,
                'kfold': kfold,
                'tensorboard':tensorboard}  # if set to 0, we dont use kfold cross validation

    if '--show_settings' in args:
        print("\nYou are using the following settings:")
        print(settings)
        sys.exit(-1)

    for idx, arg in enumerate(args):
        if '--' in arg:  # this is a key, value following afterwards
            arg_name = arg[2:]  # remove the two lines before the actual name
            if arg_name == 'train_only_bias':  # only train the bias parameters
                if len(args) > (idx + 1):
                    value = args[idx + 1]
                    if '--' in value:
                        settings[arg_name] = 'all'
                        continue
                    else:
                        val = str(value).lower()
                        settings[arg_name] = val
                        args.pop(idx+1)
                        continue
                else: 
                    settings[arg_name] = 'all'
            elif arg_name == 'empathy' or arg_name == 'distress':
                settings['empathy_type'] = arg_name
                continue
            elif arg_name == 'scheduler':
                settings[arg_name] = True
                continue
            elif arg_name == 'tensorboard':
                settings[arg_name] = True
                continue
            elif arg_name == 'save_settings':
                settings[arg_name] = True
                continue
            elif arg_name == 'save_model':
                settings[arg_name] = True
                continue
            elif arg_name == 'early_stopping':
                settings[arg_name] = True
                continue
            elif len(args) > (idx + 1):
                if arg_name not in settings.keys():
                    print(f"--- MyWarning: Argument ({arg_name}) not found in settings. Your setting might not me recignized or used. ---")
                    print(f"The following are possible: {list(settings.keys())}")
                    
                value = args[idx + 1]
                if '--' in value:
                    print(f"No value specified for {arg_name}. Don't use for settings. Abort.")
                else:
                    if arg_name == 'model_name':
                        settings["model_name"] = arg_name
                        settings[arg_name] = str(value)
                        args.pop(idx+1)
                        continue
                    try:
                        val = float(value)
                    except:
                        val = str(value)
                    settings[arg_name] = val
                    args.pop(idx+1)
            else:
                print(f'MyWarning: Could not recognize argument {arg}')
                continue
        else:
            print(f'MyWarning: Could not recognize argument ({arg}). Maybe try using -- before. Exiting.')
            sys.exit(-1)

            continue
        
    settings['epochs'] = int(settings['epochs'])
    settings['batch_size'] = int(settings['batch_size'])
    settings['kfold'] = int(settings['kfold'])
    settings['seed'] = int(settings['seed'])

    
    print('\n------------ Model name: ' + settings['model_name'] + ' ------------\n')
    if settings["save_settings"]:
        with open('output/settings_' + settings['model_name'] + '.json', 'w') as fp:
            json.dump(settings, fp)
    print('\nYou are using the following settings:')
    print(settings, '\n')
    return settings
