#!/usr/bin/env python
# coding: utf-8

# # Test Machine learning algorithms - No hyperparameter tuning

# In[1]:


# Import librairies

import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix,classification_report
from yellowbrick.text import TSNEVisualizer


import pickle
import shutil
import sys
import warnings

from utils_text_processing import *

# In[2]:


nltk.download("words")
nltk.download("stopwords")
nltk.download("omw-1.4")
nlp = spacy.load("fr_core_news_md")


# In[3]:


# Set paths
path = "."
os.chdir(path)
data_path = path + "/data"
output_path = path + "/outputs"
fig_path = path + "/figures"


# In[4]:


# Suppression des FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore")


# In[5]:


# Activation PEP8
#get_ipython().run_line_magic('load_ext', 'pycodestyle_magic')
#get_ipython().run_line_magic('pycodestyle_on', '')


# In[6]:


# Paramètres graphiques
#get_ipython().run_line_magic('matplotlib', 'inline')
rc = {
    'font.size': 14,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.max_open_warning': 30}

sns.set(font='Arial', rc=rc)
sns.set_style(
    "whitegrid", {
        'axes.edgecolor': 'k',
        'axes.linewidth': 1,
        'axes.grid': True,
        'xtick.major.width': 1,
        'ytick.major.width': 1
        })
sns.set_context(
    "notebook",
    font_scale=1.1,
    rc={"lines.linewidth": 1.5})

pd.set_option('display.max_columns', None)


# In[7]:


# Import data
df = pd.read_csv(os.path.join(data_path, 'working_data_rameau.csv'), index_col=0)
print(df.shape)
df.head()


# In[8]:


# add words
add_words = [
        "la",
        "de",
        "le",
        "les",
        "l",
        "au",
        "du",
        "ouvrage",
        "auteur",
        "livre",
        "quatrieme",
        "couv"
]


# In[9]:


# Select sample of data
n_sample = 50000
df_sample = df.sample(n=n_sample)
df_sample.shape


# In[10]:


# Preproces des résumés
df_sample['DESCR_processed'] = df_sample['DESCR'].apply(
    lambda x: preprocess_text(
        x,
        add_words=add_words,
        numeric=False,
        stopw=True,
        stem=False,
        lem=True))


# In[ ]:


# Split data
y = df_sample["rameau_list_unstack"]
X = df_sample["DESCR_processed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[ ]:


# Check size
print(f"train dataset size : {len(y_train)}")
print(f"test dataset size : {len(y_test)}")


# In[ ]:


# Convert the categorical labels to Multi Label Encodings
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(y_train)
test_labels = mlb.transform(y_test)


# In[ ]:


# Create TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

vectorizer = TfidfVectorizer()
vectorised_train_documents = vectorizer.fit_transform(X_train)
vectorised_test_documents = vectorizer.transform(X_test)


# In[ ]:


# Visualize Word Frequency Distribution
from yellowbrick.text import FreqDistVisualizer
plt.figure(figsize=(20, 10))
features = vectorizer.get_feature_names_out()
visualizer = FreqDistVisualizer(features=features, n=100, orient="v")
visualizer.fit(vectorised_train_documents)
visualizer.show()


# In[ ]:


# Visualize the dataset with T-SNE
tsne = TSNEVisualizer()
tsne.fit(vectorised_train_documents)
tsne.show()


# In[ ]:


# Train and Evaluate Classifiers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, hamming_loss, coverage_error, brier_score_loss,
    label_ranking_average_precision_score, label_ranking_loss)


ModelsPerformance = {}

def metricsReport(modelName, test_labels, predictions, compare="macro_f1"):

    accuracy = accuracy_score(test_labels, predictions)

    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')

    hamLoss = hamming_loss(test_labels, predictions)
    lrap = label_ranking_average_precision_score(test_labels, predictions)
    lrl = label_ranking_loss(test_labels, predictions)
    cov_error = coverage_error(test_labels, predictions)
    EMR = np.all(test_labels == predictions, axis=1).mean()
    #brier = brier_score_loss(test_labels, predictions)

    # Print result
    print("------" + modelName + " Model Metrics-----")
    print(f"Accuracy: {accuracy:.4f}\nHamming Loss: {hamLoss:.4f}\nCoverage Error: {cov_error:.4f}")
    #print(f"Brier Score: {brier:.4f}")
    print(f"Exact Match Ratio: {EMR:.4f}\nRanking Loss: {lrl:.4f}\nLabel Ranking avarge precision (LRAP): {lrap:.4f}")
    print(f"Precision:\n  - Macro: {macro_precision:.4f}\n  - Micro: {micro_precision:.4f}")
    print(f"Recall:\n  - Macro: {macro_recall:.4f}\n  - Micro: {micro_recall:.4f}")
    print(f"F1-measure:\n  - Macro: {macro_f1:.4f}\n  - Micro: {micro_f1:.4f}")

    # Store F1
    ModelsPerformance[modelName] = eval(compare)


# In[ ]:


# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

knnClf = KNeighborsClassifier()

knnClf.fit(vectorised_train_documents, train_labels)
knnPredictions = knnClf.predict(vectorised_test_documents)
metricsReport("knn", test_labels, knnPredictions)


# In[ ]:


# Decision tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtClassifier = DecisionTreeClassifier()
dtClassifier.fit(vectorised_train_documents, train_labels)
dtPreds = dtClassifier.predict(vectorised_test_documents)
metricsReport("Decision Tree", test_labels, dtPreds)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(n_jobs=-1)
rfClassifier.fit(vectorised_train_documents, train_labels)
rfPreds = rfClassifier.predict(vectorised_test_documents)
metricsReport("Random Forest", test_labels, rfPreds)


# In[ ]:


# Bagging
from sklearn.ensemble import BaggingClassifier

bagClassifier = OneVsRestClassifier(BaggingClassifier(n_jobs=-1))
bagClassifier.fit(vectorised_train_documents, train_labels)
bagPreds = bagClassifier.predict(vectorised_test_documents)
metricsReport("Bagging", test_labels, bagPreds)


# In[ ]:


# Boosting
from sklearn.ensemble import GradientBoostingClassifier

boostClassifier = OneVsRestClassifier(GradientBoostingClassifier())
boostClassifier.fit(vectorised_train_documents, train_labels)
boostPreds = boostClassifier.predict(vectorised_test_documents)
metricsReport("Boosting", test_labels, boostPreds)


# In[ ]:


# Naive Bayes Classifierf
from sklearn.naive_bayes import MultinomialNB

nbClassifier = OneVsRestClassifier(MultinomialNB())
nbClassifier.fit(vectorised_train_documents, train_labels)
nbPreds = nbClassifier.predict(vectorised_test_documents)
metricsReport("Multinomial NB", test_labels, nbPreds)


# In[ ]:


# Support Vector Machine (Linear SVC)
from sklearn.svm import LinearSVC

svmClassifier = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svmClassifier.fit(vectorised_train_documents, train_labels)

svmPreds = svmClassifier.predict(vectorised_test_documents)
metricsReport("SVC Sq. Hinge Loss", test_labels, svmPreds)


# # Binary relevance
# from sklearn.svm import LinearSVC
# from skmultilearn.problem_transform import BinaryRelevance
# 
# BinaryRelSVC = BinaryRelevance(LinearSVC())
# BinaryRelSVC.fit(vectorised_train_documents, train_labels)
# 
# BinaryRelSVCPreds = BinaryRelSVC.predict(vectorised_test_documents)

# In[ ]:


# Label powerset
from skmultilearn.problem_transform import LabelPowerset

powerSetSVC = LabelPowerset(LinearSVC())
powerSetSVC.fit(vectorised_train_documents, train_labels)

powerSetSVCPreds = powerSetSVC.predict(vectorised_test_documents)
metricsReport("Power Set SVC", test_labels, powerSetSVCPreds)


# In[ ]:


# Comparison on different models based on their Micro-F1 score
print("  Model Name " + " "*10 + "| Micro-F1 Score")
print("-------------------------------------------")
for key, value in ModelsPerformance.items():
    print("  " + key, " "*(20-len(key)) + "|", value)
    print("-------------------------------------------")

