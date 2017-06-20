from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import time

import numpy as np


"""Utility functions for parsing and writing data to file.

@author: gmeanti
"""

SUBMIT_FOLDER   = "../saved_data/submissions"
TRAIN_DATA_FILE = "../data_train.csv"
MASK_DATA_FILE  = "../sampleSubmission.csv"

MOVIELENS_DATA_FILE = "../saved_data/movielens/ml-1m/ratings_clean.csv"

def get_time():
    return time.strftime("%H:%M:%S", time.gmtime())


def read_data(use_mask=False, test_percentage=0.2, dataset="CF"):
    """Read necessary data files.
    @param use_mask: if true will read the mask for making submissions to the server.
            Otherwise will use training data for validation (amount specified
            with `test_percentage` parameter)
    @param dataset: string specifying which dataset to use. 2 possibilities are
            "movielens" which uses the movielens 1m dataset, or "CF" which uses the
            in-class dataset.
    """
    if dataset.lower() == "cf":
        if use_mask:
            M = _read_mask(MASK_DATA_FILE, 10000, 1000)
            A = _read_data(TRAIN_DATA_FILE, 0, 10000, 1000)
            return (A, None, M)
        else:
            (Atr, Ats, Mts) = _read_data(TRAIN_DATA_FILE, test_percentage, 10000, 1000)
            return (Atr, Ats, Mts)
    elif dataset.lower() == "movielens":
        (Atr, Ats, Mts) = _read_data(MOVIELENS_DATA_FILE, test_percentage, 6040, 3952)
        return (Atr, Ats, Mts)
    else:
        print("%s dataset %s is not implemented" % (get_time(), dataset))
        return None # TODO: throw exception

def write_predict(pfunc, M, file_name):
    """Writes the predictions to a file, using as test points
    those specified in the mask, and `pfunc` to ask for predictions.
    `pfunc` must take 2 arguments: user and movie indices
    """
    count = 0
    with open(file_name, 'w') as fh:
        fh.write("Id,Prediction\n")
        for x in range(M.shape[0]):
            for y in range(M.shape[1]):
                if M[x, y] == True:
                    count += 1
                    fh.write("r%d_c%d,%f\n" % (x+1, y+1, pfunc(x, y)))
    print("Written %d predictions to %s" % (count, file_name))


def __extract_line_data(l):
    us = l.index('_')
    row = int(l[1:us])-1
    com = l.index(',')
    col = int(l[us+2:com])-1
    val = int(l[com+1:l.index('\n')])
    return (row, col, val)


def _read_data(file_name, test_percentage, nusers, nitems):
    matrix_size = (nusers, nitems)
    Atr = np.zeros(matrix_size, dtype="int")
    
    with open(file_name, 'r') as fh:
        data = fh.readlines()
    data_size = len(data) - 1
    
    if test_percentage > 0:
        Ats = np.zeros(matrix_size, dtype="int")
        Mts = np.zeros(matrix_size, dtype="bool")
        prob_test = np.random.rand(data_size)

    for i, l in enumerate(data[1:]):
        row, col, val = __extract_line_data(l)
        if test_percentage > 0 and prob_test[i] < test_percentage:
            Ats[row, col] = val
            Mts[row, col] = True
        else:
            Atr[row, col] = val
    if test_percentage > 0:
        return (Atr, Ats, Mts)
    return Atr


def _read_mask(file_name, nusers, nitems):
    matrix_size = (nusers, nitems)
    with open(file_name, 'r') as fh:
        data = fh.readlines()
    A = np.zeros(matrix_size, dtype="bool")
    for l in data[1:]:
        row, col, _ = __extract_line_data(l)
        A[row, col] = True
    return A
