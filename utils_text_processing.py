"""Utilitary functions used for text processing in Project 6"""

# Import des librairies
import os
import re

import nltk
import numpy as np
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize



# download nltk packages
nltk.download("words")
nltk.download("stopwords")
nltk.download("omw-1.4")
nlp = spacy.load("fr_core_news_md")

DPI = 300
RAND_STATE = 42

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_module_url = "resources/USE"
use_model = hub.load(use_module_url)
print("USE model %s loaded")


#                           TEXT PREPROCESS                         #
# --------------------------------------------------------------------
def flatten(l):
    return [item for sublist in l for item in sublist]


# Remove stopwords
def stop_word_filter_fct(list_words, add_words):
    """
    Description: remove classical french (and optionnally more)
    stopword from a list of words

    Arguments:
        - list_words (list): list of words
        - add_words (list) : list of additionnal words to remove

    Returns :
        - text without stopwords
    """
    stop_w = list(set(stopwords.words("french"))) + add_words
    filtered_w = [w for w in list_words if w not in stop_w]
    return filtered_w


# Lemmatizer (base d'un mot)
def lemma_fct(list_words):
    """
    Description: lemmatize a list of words

    Arguments:
        - list_words (list): list of words
        - lang (str): language used in the corpus (default: english)

    Returns :
        - Lemmatized list of words
    """
    # if lang == "english":
    #     lemma = WordNetLemmatizer()
    #     lem_w = [
    #         lemma.lemmatize(
    #             lemma.lemmatize(
    #                 lemma.lemmatize(lemma.lemmatize(w, pos="a"), pos="v"), pos="n"
    #             ),
    #             pos="r",
    #         )
    #         for w in list_words
    #     ]

    lem_w = []
    for token in nlp(" ".join(list_words)):
        lem_w.append(token.lemma_)

    # lem_w = " ".join(map(str, empty_list))

    return lem_w


# Stemming
def stem_fct(list_words):
    """
    Description: Stem a list of words

    Arguments:
        - list_words (list): list of words

    Returns :
        - Stemmed list of words
    """
    stemmer = FrenchStemmer()
    stem_w = [stemmer.stem(w) for w in list_words]

    return stem_w


# Preprocess text
def preprocess_text(text, add_words=[], numeric=True, stopw=True, stem=False, lem=True):
    """
    Description: preprocess a text with different preprocessings.

    Arguments:
        - text (str): text, with punctuation
        - add_words (str): words to remove, in addition to classical english stopwords
        - numeric (bool): whether to remove numerical or not (default: True)
        - stopw (bool): whether to remove classical english stopwords (default: True)
        - stem (bool): whether to stem words or not (default: False)
        - lem (bool): whether to lemmatize words or not (default: True)
        - lang (str): language used in the corpus (default: eng for english). Can be 'eng' or 'fr'. 

    Returns :
        - Preprocessed list of tokens
    """

    # Lowerize all words
    text_lower = text.lower()

    # remove particular characters and punctuation
    text_lower = re.sub(r"_", " ", text_lower)  # r"_| x |\d+x"
    text_no_punct = re.sub(r"[^\w\s]", " ", text_lower)

    if numeric:
        # remove all numeric characters
        text_no_punct = re.sub(r"[^\D]", " ", text_no_punct)

    # tokenize
    word_tokens = word_tokenize(text_no_punct)

    if stopw:
        # remove stopwords
        word_tokens = stop_word_filter_fct(word_tokens, add_words)

    # Stemming or lemmatization
    if stem:
        # stemming
        word_tokens = stem_fct(word_tokens)

    if lem:
        # lemmatization
        word_tokens = lemma_fct(word_tokens)

    # if stopw:
    # remove stopwords
    # word_tokens = stop_word_filter_fct(word_tokens, add_words)

    # Finalize text
    transf_desc_text = " ".join(word_tokens)

    return transf_desc_text