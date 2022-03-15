import pandas as pd

def load_data(data_root_folder="../data/"):
    # dev and train data
    data_train = pd.read_csv(data_root_folder + "messages_train_ready_for_WS.tsv", sep='\t')
    features_dev = pd.read_csv(data_root_folder + "messages_dev_features_ready_for_WS_2022.tsv", sep='\t')
    labels_dev = pd.read_csv(data_root_folder + "goldstandard_dev_2022.tsv", sep='\t', header=None)
    return data_train, features_dev, labels_dev


def load_sentenced_emotions(data_root_folder="../data/"):
    # sentencized automatic emotion tags
    sent_emotion_train = pd.read_csv(data_root_folder + "messages_train_sentencized_automatic_emotion_tags.tsv", sep='\t')
    sent_emotion_dev = pd.read_csv(data_root_folder + "messages_dev_sentencized_automatic_emotion_tags.tsv", sep='\t')
    return sent_emotion_train, sent_emotion_dev


def load_articles(data_root_folder="../data/"):
    # the news articles
    articles = pd.read_csv(data_root_folder + "articles_adobe_AMT.csv")
    return articles
