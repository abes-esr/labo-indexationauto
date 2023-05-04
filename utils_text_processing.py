""" Utilitary functions for text preprocessing"""

# Import librairies
import os
import re


import nltk
import numpy as np
import pandas as pd
import unicodedata
import spacy
import shutil
import sys
import fr_core_news_sm
import en_core_web_sm

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from utils_visualization import *
from utils_model_optimization import *

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop


# Test processing


nltk.download("punkt")
nltk.download("words")
nltk.download("stopwords")
nltk.download("omw-1.4")

# Set paths
path = "."
os.chdir(path)
data_path = path + "/data"
output_path = path + "/outputs"
fig_path = path + "/figs"


# Set Parameters
RANDOM_STATE = 42


# Import dataset
def get_dataset(filename):
    dataset = pd.read_csv(
        os.path.join(data_path, filename),
        converters={"DESCR": eval,  "rameau_concepts": eval},
    )
    return dataset


# Save Dataset to csv
def save_dataset_to_csv(df, filename):
    path_destination = os.path.join(data_path, filename)
    df.to_csv(path_destination, index=False)


# Import processed dataset
def get_dataset_init(filename):
    dataset_init = pd.read_csv(
        os.path.join(data_path, filename), converters={"rameau_concepts": eval}
    )
    return dataset_init


#                           TEXT PREPROCESS                         #
# --------------------------------------------------------------------
def flatten(l):
    return [item for sublist in l for item in sublist]


class PreprocessData:
    def __init__(
        self,
        df,
        input_col,
        output_col,
        lang="french",
        add_words=[],
        ascii=False,
        numeric=True,
        stopw=True,
        stem=False,
        lem=True,
    ):
        self.df = df
        self.lang = lang
        self.add_words = add_words
        self.ascii = ascii
        self.numeric = numeric
        self.stopw = stopw
        self.stem = stem
        self.lem = lem

        if lang == "french":
            self.stop_w = set(stopwords.words(lang)).union(fr_stop).union(add_words)
            ref = "fr_core_news_md"
        elif lang == "english":
            self.stop_w = set(stopwords.words(lang)).union(en_stop).union(add_words)
            ref = "en_core_web_sm"
        else:
            ValueError(
                f"Unknown language, must be 'french' or 'english', you provided {lang}"
            )
        self.nlp = spacy.load(ref, disable=["parser", "ner"])
        self.stemmer = SnowballStemmer(language=lang)

        df_copy = df.copy()
        df_copy[output_col] = df_copy[input_col].apply(
            lambda x: self.preprocess_text(x)
        )
        self.df = df_copy

    def stop_word_filter_fct(self, tokens):
        """
        Description: remove classical french (and optionnally more)
        stopword from a list of words

        Arguments:
            - list_words (list): list of words
            - add_words (list) : list of additionnal words to remove

        Returns :
            - text without stopwords
        """

        stop_w = self.stop_w
        filtered_tokens = [w for w in tokens if w not in stop_w]

        return filtered_tokens

    def funcEnc(self, tokens):
        new_words = []
        for word in tokens:
            new_word = (
                unicodedata.normalize("NFKD", word)
                .encode("ASCII", "ignore")
                .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        return new_words

    def lemma_fct(self, tokens):
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

        nlp = self.nlp

        lem_w = []
        for token in nlp(" ".join(tokens)):
            lem_w.append(token.lemma_)

        return lem_w

    def stem_fct(self, tokens):
        """
        Description: Stem a list of words

        Arguments:
            - list_words (list): list of words

        Returns :
            - Stemmed list of words
        """
        stemmer = self.stemmer
        stem_w = [stemmer.stem(w) for w in tokens]

        return stem_w

    def preprocess_text(self, text):
        """
        Description: preprocess a text with different preprocessings.

        Arguments:
            - text (str): text, with punctuation
            - add_words (str): words to remove, in addition to classical english stopwords
            - ascii (bool): whether to transform text into ascii standard (default: False)
            - numeric (bool): whether to remove numerical or not (default: True)
            - stopw (bool): whether to remove classical english stopwords (default: True)
            - stem (bool): whether to stem words or not (default: False)
            - lem (bool): whether to lemmatize words or not (default: True)
            - lang (str): language used in the corpus (default: eng for english). Can be 'eng' or 'fr'.

        Returns :
            - Preprocessed list of tokens
        """

        # Lowerize all words
        text_lower = str.lower(text)

        # remove particular characters and punctuation
        text_lower = re.sub(r"_", " ", text_lower)  # r"_| x |\d+x"
        text_no_punct = re.sub(r"[^\w\s]", " ", text_lower)

        if self.numeric:
            # remove all numeric characters
            text_no_punct = re.sub(r"[^\D]", " ", text_no_punct)

        # tokenize
        tokens = word_tokenize(text_no_punct, language=self.lang)

        if self.ascii:
            tokens = self.funcEnc(tokens)

        if self.stopw:
            # remove stopwords
            tokens = self.stop_word_filter_fct(tokens)

        # Stemming or lemmatization
        if self.stem:
            # stemming
            tokens = self.stem_fct(tokens)

        if self.lem:
            # lemmatization
            tokens = self.lemma_fct(tokens)

        # if stopw:
        # remove stopwords
        # word_tokens = stop_word_filter_fct(word_tokens, add_words)

        # Finalize text
        text = " ".join(tokens)
        self.tokens = tokens
        return text


# Remove stopwords
def stop_word_filter_fct2(tokens, lang="french", add_words=None):
    """
    Description: remove classical french (and optionnally more)
    stopword from a list of words

    Arguments:
        - list_words (list): list of words
        - add_words (list) : list of additionnal words to remove

    Returns :
        - text without stopwords
    """
    if lang == "french":
        stop_w = set(stopwords.words(lang)).union(fr_stop).union(add_words)
    elif lang == "english":
        stop_w = set(stopwords.words(lang)).union(en_stop).union(add_words)
    else:
        ValueError(
            f"Unknown language, must be 'french' or 'english', you provided {lang}"
        )
    filtered_w = [w for w in tokens if w not in stop_w]
    return filtered_w


# Encodage des données, les accents seront enlevés
def funcEnc2(tokens):
    new_words = []
    for word in tokens:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ASCII", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words


# Lemmatizer (base d'un mot)
def lemma_fct2(list_words, lang="french"):
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
    if lang == "french":
        ref = "fr_core_news_md"
    if lang == "english":
        ref = "en_core_web_sm"

    nlp = spacy.load(ref, disable=["parser", "ner"])
    for token in nlp(" ".join(list_words)):
        lem_w.append(token.lemma_)

    return lem_w


# Stemming
def stem_fct2(list_words, lang="french"):
    """
    Description: Stem a list of words

    Arguments:
        - list_words (list): list of words

    Returns :
        - Stemmed list of words
    """
    stemmer = SnowballStemmer(language=lang)
    stem_w = [stemmer.stem(w) for w in list_words]

    return stem_w


# Preprocess text
def preprocess_text2(
    text,
    lang="french",
    add_words=[],
    ascii=False,
    numeric=True,
    stopw=True,
    stem=False,
    lem=True,
):
    """
    Description: preprocess a text with different preprocessings.

    Arguments:
        - text (str): text, with punctuation
        - add_words (str): words to remove, in addition to classical english stopwords
        - ascii (bool): whether to transform text into ascii standard (default: False)
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

    if ascii:
        word_tokens = funcEnc2(word_tokens)

    if stopw:
        # remove stopwords
        word_tokens = stop_word_filter_fct2(word_tokens, lang, add_words)

    # Stemming or lemmatization
    if stem:
        # stemming
        word_tokens = stem_fct2(word_tokens, lang)

    if lem:
        # lemmatization
        word_tokens = lemma_fct2(word_tokens, lang)

    # if stopw:
    # remove stopwords
    # word_tokens = stop_word_filter_fct(word_tokens, add_words)

    # Finalize text
    transf_desc_text = " ".join(word_tokens)

    return transf_desc_text


#                           TEXT VECTORIZATION                       #
# --------------------------------------------------------------------
def convert_bow_tfidf(
    method,
    fit_on,
    transform_on=None,
    max_df=1,
    min_df=1,
    n_gram=(1, 1),
    max_features=None,
):
    """
    Description: convert a collection of raw documents to document terms matrix
    Arguments:
        - method : count method (CountVectorizer or TfidfVectorizer)
        - fit_on (pd.dataframe): data to fit on
        - transform_on (pd.dataframe): data to transform.
            If not set, fit and transform on the same data
        - max_df (float): ignore terms that have a document frequency strictly
            higher than max_df
        - min_df (float): ignore terms that have a document frequency strictly
            lower than min_df
        - n_gram (tuple): lower and upper boundary of the range of n-values
            for different word n-grams or char n-grams to be extracted
        - max_features (int): build a vocabulary that only consider
            the top max_features ordered by term frequency across the corpus
    Return :
        - transformed data, ready to use for dimension reduction or clustering
    """
    # set model
    model = method(
        max_df=max_df, min_df=min_df, ngram_range=n_gram, max_features=max_features
    )

    # fit
    model.fit(fit_on)

    # predict
    if transform_on:
        cv_transform = model.transform(transform_on)
    else:
        cv_transform = model.transform(fit_on)

    return cv_transform


def feature_word2vec_fct(
    sentences,
    min_count=3,
    window=3,
    vector_size=30,
    seed=RAND_STATE,
    workers=1,
    skig_gram=0,
    w2v_epochs=100,
    n_words=100,
    plot=False,
):
    """
    Description: compute word2vec embedding, using gensim implementation.
        Parameters are the same than gensim word2vec embedding
    Arguments:
        - sentences (list): list of lists of tokens
        - min_count (int): ignores all words with total frequency lower than min_count.
        - window (int): Maximum distance between the current and predicted
            word within a sentence
        - vector_size (int): dimensionality of the word vectors
        - seed (int): seed for the random number generator
        - workers (int): use these many worker threads to train the model
            (=faster training with multicore machines)
        - skig_gram ({0, 1}): training algorithm: 1 for skip-gram; otherwise CBOW.
        - w2v_epochs (int): number of iterations (epochs) over the corpus
        - plot (bool): whether to plot visualisation of embedding
        - n_words (int) : number of words to plot if plotting asked
    Return :
        - embed data, ready to use for dimension reduction or clustering
    """

    # Création et entraînement du modèle
    print("Build & train Word2Vec model ...")
    w2v_model = Word2Vec(
        min_count=min_count,
        window=window,
        vector_size=vector_size,
        seed=seed,
        workers=workers,
        sg=skig_gram,
        epochs=w2v_epochs,
    )

    w2v_model.build_vocab(sentences)
    w2v_model.train(
        sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs
    )

    # Récupération des valeurs
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key

    print("Vocabulary size: %i" % len(w2v_words))
    print("Word2Vec trained")

    # Vectorisation des inputs
    print("Vectorize documents ...")
    vectorized_docs = vectorize(sentences, model=w2v_model)

    if plot:
        visualize_w2v(w2v_model, n_words=n_words)

    return vectorized_docs


def feature_doc2vec_fct(
    sentences,
    min_count=3,
    window=3,
    vector_size=30,
    seed=RAND_STATE,
    workers=1,
    w2v_epochs=100,
):
    """
    Description: compute doc2vec embedding, using gensim implementation.
        Parameters are the same than gensim word2vec embedding
    Arguments:
        - sentences (list): list of lists of tokens
        - min_count (int): ignores all words with total frequency
            lower than min_count.
        - window (int): Maximum distance between the current and predicted
            word within a sentence
        - vector_size (int): dimensionality of the word vectors
        - seed (int): seed for the random number generator
        - workers (int): use these many worker threads to train the model
            (=faster training with multicore machines)
        - w2v_epochs (int): number of iterations (epochs) over the corpus
    Return :
        - embed data, ready to use for dimension reduction or clustering
    """

    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences)]
    print("Build & train Doc2Vec model ...")
    d2v_model = Doc2Vec(
        min_count=min_count,
        window=window,
        vector_size=vector_size,
        seed=seed,
        workers=workers,
        epochs=w2v_epochs,
    )

    d2v_model.build_vocab(tagged_data)
    d2v_model.train(
        tagged_data, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs
    )

    # Récupération des valeurs
    model_vectors = d2v_model.dv
    w2v_docs = model_vectors.index_to_key
    print("Corpus size: %i" % len(w2v_docs))
    print("Doc2Vec trained")

    # Vectorisation des inputs
    print("Vectorize documents ...")
    vectorized_docs = vectorize(sentences, model=d2v_model)

    return vectorized_docs


def vectorize(list_of_docs, model):
    """
    Description: Generate vectors for list of documents using a Word Embedding

    Arguments:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)

    return features


def explore_preprocess_vector_dim(
    df,
    categorical_labels,
    preprocess_list,
    vectorization_method_list,
    red_dim_method_list,
    clustering_method,
    params,
    summary,
    summary_filename,
):
    """
    Description: compute end to end matrix terms frequency/embedding methods,
        reduction dimension, clustering and metric computation for all combinations
        of preprocess, vectorization methods, and reduction dimension.
        Results are stored in a dataframe.

    Arguments:
        - df (pd.DataFrame): dataframe containing data
        - categorical_labels (np.array1D): true labels of all df rows
        - preprocess_list (np.array1D): list of preprocessed data,
            must be columns names of df
        - vectorization_method_list (np.array1D): list of vectorization methods.
            Can be :
                - CountVectorizer or TfidfVectorizer
                - Word2Vec or Doc2Vec
                - 'BERT' or 'USE'
        - red_dim_method_list (np.array1D): list of methods for dimension reduction.
            Can be :
                - 'No reduction', 'TSNE', 'NMF',
                - 'PCA/LSA' for PCA/TruncatedSVD,
                - 'LDA' for LatentDirichletAllocation, ...)
        - clustering_method (clusterMixin) : clustering method (e.g: KMeans)
        - params (dict): dictionnary containing all parameters for vectorization
            and reduction dimension methods.
        - summary (pd.DataFrame) : summary dataframe to store results
        - summary_filename (str) : name of the csv filename to save summary dataframe
    Returns:
        Summary of parameters and metrics (ARI, silhouette score and execution time)
            for all combinations of preprocess, vectorization and dimension reduction methods.
    """

    for method in vectorization_method_list:
        if hasattr(method, "__name__"):
            print(
                str(
                    "####################   Vectorization method:  "
                    + method.__name__
                    + "####################"
                )
            )
        else:
            print(
                str(
                    "####################   Vectorization method:  "
                    + str(method)
                    + "####################"
                )
            )

        for feature in preprocess_list:
            print("FEATURE:  ", feature)
            print("--------------------------------")

            time1 = time.time()

            # Vectorization
            if method == CountVectorizer or method == TfidfVectorizer:
                preprocess = "bag_of_word/tf-idf"
                method_for_summary = method
                # conversion en une matrice sparse
                cv_transform = convert_bow_tfidf(
                    method,
                    fit_on=df[feature],
                    max_df=params["bow_max_df"],
                    min_df=params["bow_min_df"],
                    n_gram=params["bow_n_gram"],
                    max_features=params["bow_max_features"],
                )
                cv_transform_scaled = cv_transform

            else:
                # Definition des phrases
                sentences = df[feature].to_list()

                if method == Word2Vec:
                    preprocess = "word2vec/doc2vec"
                    method_for_summary = method
                    cv_transform = feature_word2vec_fct(
                        sentences=sentences,
                        min_count=params["w2v_min_count"],
                        window=params["w2v_window"],
                        vector_size=params["w2v_size"],
                        seed=params["rand_state"],
                        workers=params["w2v_workers"],
                        skig_gram=params["w2v_skig_gram"],
                        w2v_epochs=params["w2v_epochs"],
                    )

                elif method == Doc2Vec:
                    preprocess = "word2vec/doc2vec"
                    method_for_summary = method
                    cv_transform = feature_doc2vec_fct(
                        sentences=sentences,
                        min_count=params["w2v_min_count"],
                        window=params["w2v_window"],
                        vector_size=params["w2v_size"],
                        seed=params["rand_state"],
                        workers=params["w2v_workers"],
                    )

                #  Scale data to get only positive values
                cv_transform_scaled = MinMaxScaler().fit_transform(cv_transform)

            for red_dim_method in red_dim_method_list:
                # Reduction dimension
                print("Reduction dimension via ", red_dim_method)

                if red_dim_method == "No reduction":
                    x_red = cv_transform_scaled
                    n_comp = None
                    explained_var = "nan"
                else:
                    x_red, explained_var, n_comp = reduce_dimension(
                        red_dim_method, cv_transform_scaled, **params["n_comp_list"]
                    )

                # Détermination des clusters à partir des données après réduction dimensions
                print("Clustering ...")
                cls = clustering_method(
                    n_clusters=params["num_labels"],
                    init=params["kmeans_init"],
                    n_init=params["kmeans_n_init"],
                    random_state=params["rand_state"],
                )
                cls.fit(x_red)
                labels = cls.labels_

                # Calcul des métriques
                print("Computing metrics ...")
                ARI, silhouette = compute_metrics(
                    x_red, categorical_labels, labels, round=4
                )
                time2 = np.round(time.time() - time1, 0)

                # Mise à jour tableau de résultats
                print("Saving results ...")
                if hasattr(cv_transform_scaled, "shape"):
                    matrix_shape = cv_transform_scaled.shape
                else:
                    matrix_shape = np.asarray(cv_transform_scaled).shape

                summary = summary.append(
                    {
                        "preprocess": preprocess,
                        "feature_name": feature,
                        "reduction_dimension": red_dim_method,
                        "n_comp": n_comp,
                        "explained_var": explained_var,
                        "algorithm": method_for_summary.__name__,
                        "matrix_dim_after_vect": matrix_shape,
                        "matrix_dim_after_reduc": np.asarray(x_red).shape,
                        "ARI": ARI,
                        "Silhouette_score": silhouette,
                        "execution_time": time2,
                        "outputs": {
                            "cv_transform": cv_transform,
                            "cv_transform_scaled": cv_transform_scaled,
                            "cls": cls,
                            "x_red": x_red,
                            "labels": labels,
                        },
                    },
                    ignore_index=True,
                )

                summary.to_csv(summary_filename)
                summary.to_pickle(str("p_" + summary_filename))

    return summary


def gridsearch_preprocess_vector_dim(
    df,
    categorical_labels,
    preprocess_list,
    vectorization_method_name_list,
    vectorization_method_list,
    red_dim_method_list,
    clustering_method,
    grid_params,
    nlp_params_range,
    summary,
    summary_filename,
    **fit_params,
):
    """
    Description: compute hyper-parameter optimization on end to end
        matrix terms frequency/embedding methods, reduction dimension,
        clustering and metric computation for all combinations
        of preprocess, vectorization methods, and reduction dimension.
        Results are stored in a dataframe.

    Arguments:
        - df (pd.DataFrame): dataframe containing data
        - categorical_labels (np.array1D): true labels of all samples present in df
        - preprocess_list (np.array1D): list of preprocessed data, must be columns names of df
        - vectorization_method_list (np.array1D): list of vectorization methods.
            Can be :
                - CountVectorizer or TfidfVectorizer
                - Word2Vec or Doc2Vec
                - 'BERT' or 'USE'
        - vectorization_method_name_list (np.array1D): name of vectorization family
            associated to each vectorization method present in vectorization_method_list.
            E.g : 'bow/tf-idf', 'word2vec/doc2vec', 'bert', 'use'
        - red_dim_method_list (np.array1D): list of methods for dimension reduction.
            Can be :
                - 'No reduction', 'TSNE', 'NMF',
                - 'PCA/LSA' for PCA/TruncatedSVD,
                - 'LDA' for LatentDirichletAllocation, ...)
        - clustering_method (clusterMixin) : clustering method (e.g: KMeans)
        - grid_params (dict): dictionnary containing shared parameters accross methods
        - nlp_params_range (dict): dictionnary containing all specific parameters
            for vectorization and reduction dimension methods.
        - summary (pd.DataFrame) : summary dataframe to store results
        - summary_filename (str) : name of the csv filename to save summary dataframe
        - fit_params (dict): other parameters necessary for gridsearchCV
    Returns:
        Summary of parameters and metrics (ARI, silhouette score and execution time)
            for best parameter optimization (based on ARI maximization)
            for all combinations of preprocess, vectorization
            and dimension reduction methods.
    """

    for feature in preprocess_list:
        print("FEATURE:  ", feature)
        print("--------------------------------")
        for vectorization, vectorization_name in zip(
            vectorization_method_list, vectorization_method_name_list
        ):
            print("                  ", str(vectorization))
            print(
                "-------------------------------------------------------------------------------"
            )
            if vectorization_name == "bag_of_word/tf-idf":
                grid_params.update(
                    {
                        "vectorizer__ngram_range": nlp_params_range["bow_n_gram"],
                        "vectorizer__max_df": nlp_params_range["bow_max_df"],
                        "vectorizer__min_df": nlp_params_range["bow_min_df"],
                        "vectorizer__max_features": nlp_params_range[
                            "bow_max_features"
                        ],
                    }
                )
            else:
                if "vectorizer__ngram_range" in grid_params:
                    grid_params.pop("vectorizer__ngram_range")
                if "vectorizer__max_df" in grid_params:
                    grid_params.pop("vectorizer__max_df")
                if "vectorizer__min_df" in grid_params:
                    grid_params.pop("vectorizer__min_df")
                if "vectorizer__max_features" in grid_params:
                    grid_params.pop("vectorizer__max_features")

            for dim_red in red_dim_method_list:
                print("Treating :", dim_red)
                print("-----------------------------")
                if str(dim_red) == "TSNE_wrapper()":
                    grid_params["dim_reductor__n_components"] = nlp_params_range[
                        "tsne_comp"
                    ]
                    grid_params.update(
                        {
                            "dim_reductor__perplexity": nlp_params_range[
                                "tsne_perplexity"
                            ],
                            "dim_reductor__init": nlp_params_range["tsne_init"],
                        }
                    )
                else:
                    if "dim_reductor__perplexity" in grid_params:
                        grid_params.pop("dim_reductor__perplexity")
                    if "dim_reductor__init" in grid_params:
                        grid_params.pop("dim_reductor__init")

                    if str(dim_red) == "PCA()":
                        grid_params["dim_reductor__n_components"] = nlp_params_range[
                            "ncomp_pca"
                        ]

                    else:
                        grid_params["dim_reductor__n_components"] = nlp_params_range[
                            "ncomp_nmf_lda"
                        ]

                pipe = Pipeline(
                    steps=[
                        ("vectorizer", vectorization),
                        ("dim_reductor", dim_red),
                        ("clustering", clustering_method),
                    ]
                )

                # Build gridsearch pipeline
                print("Gridsearch in progress ... ")
                gridPipeline = gridsearch_pipe(
                    estimator=pipe, grid_params=grid_params, **fit_params
                )
                # Fit
                gridPipeline.fit(df[feature], categorical_labels)

                # Recupère les résultat
                best_res = get_modelCV_output(gridPipeline)
                best_res
                print("ARI :", best_res.mean_test_score)

                print("Update summary file")
                summary = update_gridsearch_summary(
                    summary,
                    feature,
                    vectorization_name,
                    vectorization,
                    dim_red,
                    best_res,
                )

                print("Save summary file")
                print("--------------------")
                summary.to_csv(summary_filename)
                summary.to_pickle(str("p_" + summary_filename))

    return summary
"""Utilitary functions used for text processing in ABES project"""

# Import des librairies
import re

import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

# download nltk packages
nltk.download("words")
nltk.download("stopwords")
nltk.download("omw-1.4")
nlp = spacy.load("fr_core_news_md")

DPI = 300
RAND_STATE = 42


#                           TEXT PREPROCESS                         #
# --------------------------------------------------------------------
def flatten(liste):
    return [item for sublist in liste for item in sublist]


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
    stop_w = list(set(stopwords.words("french"))) + list(fr_stop) + add_words
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
