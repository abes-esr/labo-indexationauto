"""Utilitary functions for vizualisation used in Project 6"""

# Import des librairies
import os

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from PIL import Image
from wordcloud import WordCloud

DPI = 300
RAND_STATE = 42


#                            EXPLORATION                             #
# --------------------------------------------------------------------
def plot_wordcloud(token_list, cat=None, figsave=None):
    """
    Description: plot wordcloud of most important tokens from a list of tokens

    Arguments:
        - token_list (list): list of token lists
        - cat (str): categorie name for plot title
        - figsave (str) : name of the figure if want to save it

    Returns :
        - Wordcloud of tokens, based on tokens counts
    """
    wc = WordCloud(background_color="white", width=1000, height=500)
    wordcloud = wc.generate_from_text(" ".join(token_list))

    plt.figure(figsize=(12, 6))
    plt.suptitle(cat, fontsize=24, fontweight="bold")
    plt.imshow(wordcloud)
    plt.axis("off")

    if figsave:
        plt.savefig(figsave, dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_barplot_of_tags(
    tags_list,
    nb_of_tags,
    xlabel="Nombre d'occurences",
    ylabel="",
    figsave=None,
    figsize=(10, 30),
):
    """
    Description: plot barplot of tags count (descending order) from a list of tags

    Arguments:
        - tags_list (lsit): list of tags
        - nb_of_tags (int) : number of tags to plot in barplot (default=50)
        - xlabel, ylabel (str): labels of the barplot
        - figsize (list) : figure size (default : (10, 30))

    Returns :
        - Barplot of nb_of_tags most important tags

    """
    tag_count = Counter(tags_list)
    tag_count_sort = dict(tag_count.most_common(nb_of_tags))

    plt.figure(figsize=figsize)
    sns.barplot(
        x=list(tag_count_sort.values()),
        y=list(tag_count_sort.keys()),
        orient="h",
        palette="viridis",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if figsave:
        plt.savefig(figsave, bbox_inches="tight")
    plt.show()