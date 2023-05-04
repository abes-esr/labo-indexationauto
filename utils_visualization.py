"""Utilitary functions for vizualisation used in ABES project"""

# Import des librairies
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

DPI = 300
RAND_STATE = 42


#                            EXPLORATION                             #
# --------------------------------------------------------------------
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