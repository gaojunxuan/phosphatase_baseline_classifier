import pickle
import os

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn import model_selection
from gpmodel import gpmodel, gpkernel, chimera_tools
import encoding_tools

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix, classification_report
from pandas import DataFrame
from sklearn.utils.fixes import loguniform
import matplotlib.pyplot as plt

activations = pd.read_csv('../data/activations.csv', index_col=0)

with open('../data/EFI_ID_List.p', 'rb') as EFI:
    EFIs = pd.Series(map(str, pickle.load(EFI)))
    
with open('../data/metabolite_dict.p', 'rb') as metabolite:
    metabolite_dict = pickle.load(metabolite)
renaming_dict = dict(zip(range(len(metabolite_dict)), metabolite_dict.values()))

with open('../data/Protein_seq_dict.p', 'rb') as Protein_seq:
    Protein_seq_dict = pickle.load(Protein_seq)
    
def select_X_and_Y(df, x_rows, y_column):
    """
    Select the given X rows and Y column
    """
    # filter out empty columns
    not_dropped = ~pd.isnull(df[y_column])
    not_dropped = pd.Series(not_dropped, index=df.index)
    Ys = df[not_dropped].loc[x_rows, y_column]
    return x_rows, Ys


def process_data(df, threshold=0, transpose=False, renaming=None):
    """
    Convert numerical absorbance data into categorical data
    (active=1, inactive=-1)
    """
    formatted = df > threshold
    if renaming:
        formatted = formatted.rename(renaming)
    if transpose:
        return formatted.transpose() * 2 - 1
    return formatted * 2 - 1


def prepare_train_test(metabolite_name, threshold=0, reshape=True):
    """
    Format the data and return the training set and testing set.
    """
    df = process_data(activations, threshold=threshold, transpose=True, renaming=renaming_dict)
    xs, ys = select_X_and_Y(df, EFIs, metabolite_name)
    
    max_len = len(max(Protein_seq_dict.values(), key=len))
    fillchar = '-' # This is whats used in the GP-UCB paper
    Padded_dict = {}
    OH_dict = {}
    for ID in EFIs:
        Padded_dict[ID] = Protein_seq_dict[int(ID)].upper().ljust(max_len, fillchar)
        OH_dict[ID] = encoding_tools.one_hot_seq(seq_input=Padded_dict[ID])

    mapped_xs = xs.map(OH_dict)
    
    if reshape:
        for i, x in enumerate(mapped_xs):
            mapped_xs[i] = np.array(x.reshape(len(x)*21))
        
    X = np.array(list(mapped_xs))
    y = np.array(ys)

    return train_test_split(X, y, test_size=0.2)

from tpot import TPOTClassifier

def tpot_fit(metabolite_name, threshold=0):
    X_train, X_test, y_train, y_test = prepare_train_test(metabolite_name, threshold=threshold)
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline_{}.py'.format(metabolite_name))
    # plot_confusion_matrix(pipeline_optimizer, X_test, y_test)

    
for i, metabolite in enumerate(list(metabolite_dict.values())[:10]):
    med = activations.transpose()[i].median()
    tpot_fit(metabolite, threshold=med)
