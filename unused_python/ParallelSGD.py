from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
try:
    from time import process_time
except ImportError:
    from time import clock as process_time


import numpy as np
from matplotlib import pyplot as plt
from os import path
import copy
import util as IOutil

from joblib import Parallel, delayed


def get_time():
    return time.strftime("%H:%M:%S", time.gmtime())


data = {}
train = None

default_settings = {
    "verbose": 5,
    "max_iter": 30,
    "calc_score": 3,
    "calc_train_score": True,
    "name": "SGD",
}

self.s = copy.copy(SGD.default_settings)
self.s.update(settings)
self.test = test
self.init_data = init_data


def run(self, predict_func, update_func, post_iter_func):
    s = self.s
    global train
    indices = np.where(train != 0)

    if s["verbose"] > 3:
        print("%s Initializing vectors" % (get_time()))
    global data
    self.init_data(data, self.s)

    if s["verbose"] > 3:
        print("%s Starting SGD" % (get_time()))

    max_iter = 10000
    start = process_time()
    Parallel(n_jobs=2, backend="multiprocessing")(
        delayed(do_single_update)(indices, predict_func, update_func, s) for i in range(max_iter))
    print("Took %.2f seconds for %d iterations" % (process_time() - start, max_iter))

def do_single_update(indices, predict_func, update_func, settings):
    iindex = np.random.randint(0, len(indices[0]))
    print("Chosen index {i}".format(i=iindex))
    u = indices[0][iindex]
    i = indices[1][iindex]
    d = {"u": u, "i": i}
    global data
    global train
    ## PREDICT. self.data is shared state
    d.update(predict_func(data, settings, d))
    ## LOSS FUNCTION
    d["err"] = train[u, i] - d["prediction"]
    ## UPDATE
    data = update_func(data, settings, d)



def predict(dd, s, pd):
    u = pd["u"]; i = pd["i"]
    pd["prediction"] = dd["global_bias"] + dd["user_bias"][u] + dd["item_bias"][i] + \
                        dd["user_vecs"][u,:].dot(dd["item_vecs"][i,].T)
    return pd


def update_data(dd, s, pd):
    """
    params: dd is the data dictionary which we want to update. It is also
                the return value
            s is the settings dictionary
            pd is the prediction dictionary (returned by the prediction 
                function
    """
    u = pd["u"]; i = pd["i"]
    dd["user_bias"][u] += s["lrate1"] * (pd["err"] - s["regl6"] * dd["user_bias"][u])
    dd["item_bias"][i] += s["lrate1"] * (pd["err"] - s["regl6"] * dd["item_bias"][i])

    dd["user_vecs"][u,:] += s["lrate2"] * (pd["err"] * dd["item_vecs"][i,:] - 
                                     s["regl7"] * dd["user_vecs"][u,:])
    dd["item_vecs"][i,:] += s["lrate2"] * (pd["err"] * dd["user_vecs"][u,:] - 
                                     s["regl7"] * dd["item_vecs"][i,:])
    return dd


def update_settings(it, s):
    """Update learning rates after each iteration
    params: it is the iteration number
            s is the settings dictionary
    """
    s["lrate1"] *= s["lrate_reduction"]
    s["lrate2"] *= s["lrate_reduction"]
    return s



def simple_SGD(train, test, settings = {}):
    s = {
        "lrate1" : 0.003,     # learning rate for bias vectors
        "lrate2" : 0.001,     # learning rate for user and item vectors
        "regl6"  : 0.003,     # regularization rate for bias vectors
        "regl7"  : 0.01,      # regularization rate for user and item vectors
        "num_factors" : 20,   # size of the feature vectors
        "lrate_reduction" : 0.91 # shrinkage factor for learning rate
    }
    s.update(settings)

    def init_data(dd, s):
        """
        params: dd is the data dictionary (empty)
                s is the settings dictionary
        """
        dd["user_vecs"] = np.random.normal(scale=1./s["num_factors"],
                                    size=(s["nusers"], s["num_factors"]))
        dd["item_vecs"] = np.random.normal(scale=1./s["num_factors"],
                                    size=(s["nitems"], s["num_factors"]))
        dd["user_bias"] = np.zeros(s["nusers"])
        dd["item_bias"] = np.zeros(s["nitems"])
        dd["global_bias"] = np.mean(train[np.where(train != 0)])
        return dd


    #sSGD = SGD(s, train, test, init_data)
    #sSGD.run(predict, update_data, update_settings)
    sSGD = ParallelSGD(s, train, test, init_data)
    sSGD.run(predict, update_data, update_settings)
    return sSGD


if __name__ == "__main__":
    """Useful params:
     - need submission?
     - type of SGD
    """

    settings = {
        "nusers" : 10000,
        "nitems" : 1000,
    }
    np.random.seed(12092)
    Atr, Ats, Msk = IOutil.read_data(use_mask=False, test_percentage=0.1)
    print("%s read data." % (get_time()))
    if Ats is None:
        Ats = Msk
    sSGD = simple_SGD(Atr, Ats, settings)
