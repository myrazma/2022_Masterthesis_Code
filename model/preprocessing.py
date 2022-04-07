import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


# Load english stop words
STOPWORDS_EN = set(stopwords.words('english'))




def lower_text(text):
    return text.lower()


def word_tokenizer(text):
    """ Tokenize text based on space and punctuation using nltk word_tokenizer

    Args:
        text (str): The text

    Returns:
        list(str): The tokenized words
    """
    return nltk.word_tokenize(text)


def remove_non_alpha(text_tok):
    return [word for word in text_tok if word.isalpha()]


def remove_stopwords(text_tok):
    """Remove (english) stopwords from a tokenized texts

    Args:
        text_tok (list(str)): Tokenized text

    Returns:
        list(str): The tokenized text without stopwords
    """
    text_processed = [word for word in text_tok if not word.lower() in STOPWORDS_EN]
    return text_processed


def text_normalization(text_tok):
    # source: https://datascience.stackexchange.com/questions/69982/data-cleaning-refining-etc
    # that is not needed here, as out textual data is quite clean
    text_tok_corrected = []
    for word in text_tok:
        segmented = seg_eng.segment(word)
        corrected = sp.correct(segmented)
        text_tok_corrected.append(corrected)
        if corrected != word:
            print(word + " -> " + corrected)
    return text_tok_corrected


# after looking at some data, i feel like we dont need this here, as the data is not that messy as it might 
# be in the research on twitter data. For us it is making things just worse i guess
text = "Trump's win was an upset but it should not of really been that uprising. Some polling sites gave him a thirty percent chance of winning, and Clinton was a historical unpopular candidate. The democracts should look within on solving there problems. they should focus on nominating a better candidate the next time around."
text = lower_text(text)
text_tok = word_tokenizer(text)
text_tok = remove_non_alpha(text_tok)
text_tok_corr = text_normalization(text_tok)