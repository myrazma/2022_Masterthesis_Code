import pandas as pd
import numpy as np
import json
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

class FeatureCreator():
    def __init__(self, data_root_folder='data/'):
        self.empathy_lex, self.distress_lex = load_empathy_distress_lexicon(data_root_folder)

    def create_lexical_feature(self, data_pd, column_name='essay_tok'):
        def emp_dis_calc_exp_word_rating_score(lexicon, essay):
            # using exponential function of ratings -> higher values will have more importance
            rating_scores = []
            for word in essay:
                rating_scores.append(np.exp(lexicon[word]))

            rating_scores_np = np.array(rating_scores)
            average_rating = sum(rating_scores_np) / len(rating_scores_np)
            return average_rating
        
        data_pd['distress_word_rating'] = data_pd[column_name].apply(lambda x: emp_dis_calc_exp_word_rating_score(self.distress_lex, x))
        data_pd['empathy_word_rating'] = data_pd[column_name].apply(lambda x: emp_dis_calc_exp_word_rating_score(self.empathy_lex, x))
        return data_pd


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
    """Preprocess data and dev data including the following steps:
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


def remove_non_alpha(text_tok):
    return [word for word in text_tok if word.isalpha()]


def tokenize_data(data, column):
    """ Tokenize text in data based on space and punctuation using nltk word_tokenizer
    created new column wiht column + suffix: '_tok'

    Args:
        data (pd.DataFrame): The data including the texts
        column (str): The name of the column holding the text

    Returns:
        pd.DataFrame: The dataframe with the tokenized texts (suffix: '_tok')
    """
    data[column + '_tok'] = data[column].apply(lambda x: nltk.word_tokenize(x))
    return data

def tokenize_single_text(text):
    """ Tokenize text in data based on space and punctuation using nltk word_tokenizer

    Args:
        data (pd.DataFrame): The data including the texts
        column (str): The name of the column holding the text

    Returns:
        pd.DataFrame: The dataframe with the tokenized texts (suffix: '_tok')
    """
    return nltk.word_tokenize(text)


def lemmatize_data(data, column):
    """Lemmatize tokenized textual data using WordNetLemmatizer from nltk

    Args:
        data (pd.DataFrame): The data including the tokenized version of the texts
        column (str): The name of the column holding the tokenized text

    Returns:
        pd.DataFrame: The dataframe with a new column (suffix: '_lem')
    """

    lemmatizer = WordNetLemmatizer()  # lemmatize
    data[column + '_lem'] = data[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return data



def remove_stopwords(text_tok):
    """Remove (english) stopwords from a tokenized texts

    Args:
        text_tok (list(str)): Tokenized text

    Returns:
        list(str): The tokenized text without stopwords
    """
    text_processed = [word for word in text_tok if not word.lower() in STOPWORDS_EN]
    return text_processed


def normalize_scores(data, input_interval):
    """Maps from desired input intervall to [0,1]

    Args:
        data (np.array): The data
        input_interval ((int,int)): _description_

    Returns:
        _type_: _description_
    """
    normalized = (data - input_interval[0]) / (input_interval[1] - input_interval[0])
    return normalized


def arg_parsing_to_settings(args, default_empathy_type = 'empathy', default_learning_rate=2e-5, default_seed=17, default_batch_size=4, default_epochs=5, default_bert_type='roberta-base', default_train_only_bias=False, default_adapter_type="pfeiffer", default_model_name="", default_save_settings=False):
    # provide default settings
    settings = {'empathy_type': default_empathy_type,'learning_rate': default_learning_rate, 'seed': default_seed, 'batch_size': default_batch_size, 'epochs': default_epochs, 'bert-type': default_bert_type, "train_only_bias": default_train_only_bias, "adapter_type": default_adapter_type, "model_name": default_model_name, "save_settings":default_save_settings}

    for idx, arg in enumerate(args):
        if '--' in arg:  # this is a key, value following afterwards
            arg_name = arg[2:]  # remove the two lines before the actual name
            if arg_name == 'train_only_bias':
                settings[arg_name] = True
                continue
            elif arg_name == 'empathy' or arg_name == 'distress':
                settings["empathy_type"] = arg_name
                continue
            elif arg_name == 'save_settings':
                settings["save_settings"] = True
                continue
            elif len(args) > (idx + 1):
                if arg_name not in settings.keys():
                    print(f'--- MyWarning: Argument ({arg_name}) not found in settings. Your setting might not me recignized or used. ---')
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
                    settings[arg_name] = float(value)
                    args.pop(idx+1)
            else:
                print(f'MyWarning: Could not recognize argument {arg}')
                continue
        else:
            print(f'MyWarning: Could not recognize argument ({arg}). Maybe try using -- before.')
            continue
        
    settings['epochs'] = int(settings['epochs'])
    settings['batch_size'] = int(settings['batch_size'])


    if settings["save_settings"]:
        with open('output/settings_' + settings['model_name'] + '.json', 'w') as fp:
            json.dump(settings, fp)
    print('\nYou are using the following settings:')
    print(settings, '\n')
    return settings
