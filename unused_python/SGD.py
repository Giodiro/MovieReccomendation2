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


def plot_scores(tr_scores, ts_scores):
    fig, ax = plt.subplots()
    ax.plot(list(range(1, len(tr_scores)+1)), tr_scores, linestyle="-", linewidth=3, label="Train")
    ax.plot(list(range(1, len(ts_scores)+1)), ts_scores, linestyle="--", linewidth=3, label="Test")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.legend()
    return ax


class SGD():
    """Stochastic Gradient Descent driver"""
    default_settings = {
        "verbose": 5,
        "max_iter": 30,
        "calc_score": 3,
        "calc_train_score": True,
        "name": "SGD",
    }

    def __init__(self, settings, train, test, init_data):
        self.s = copy.copy(SGD.default_settings)
        self.s.update(settings)
        self.train = train
        self.test = test
        self.init_data = init_data
        self.data = {}

    def run(self, predict_func, update_func, post_iter_func):
        s = self.s
        indices = np.where(self.train != 0)
        shuf_indices = list(range(len(indices[0])))
        np.random.shuffle(shuf_indices)
        if s["verbose"] > 3:
            print("%s Initializing vectors" % (get_time()))
        self.init_data(self.data, self.s)
        if s["verbose"] > 3:
            print("%s Starting SGD" % (get_time()))
        self.run_internal(indices, shuf_indices, predict_func, update_func,
                          post_iter_func)
        if s["verbose"] > 3:
            print("%s Finished SGD. Obtaining predictor." % (get_time()))
        self.predictors = self.get_predictor(predict_func)


    def run_internal(self, indices, order, predict_func, update_func, post_iter_func):
        s = self.s
        tr_scores = []; ts_scores = []
        prev_score = None
        for iteration in range(s["max_iter"]):
            start = process_time()
            for q, iindex in enumerate(order):
                u = indices[0][iindex]
                i = indices[1][iindex]
                d = {"u": u, "i": i}
                #### PREDICT
                d.update(predict_func(self.data, self.s, d))
                ####
                d["err"] = self.train[u, i] - d["prediction"]
                #### UPDATE
                self.data = update_func(self.data, self.s, d)
                ####
            it_time = process_time() - start
            if iteration % s["calc_score"] == 0:
                tr_score = np.nan
                if s["calc_train_score"]:
                    tr_score = self.calc_score(self.train, predict_func)
                    tr_scores.append((iteration, tr_score))
                if self.test is not None and not self.test.dtype == bool:
                    ts_score = self.calc_score(self.test, predict_func)
                    ts_scores.append((iteration, ts_score))
                    stop_score = ts_score
                else:
                    stop_score = tr_score
                    ts_score = np.nan
                score_time = process_time() - start - it_time
                print(("{now} Iteration {i} took {t1:.2f} seconds ({t2:.2f} scoring). " + 
                       "Training score: {s1:.7f} - Test score: {s2:.7f}").format(
                          now=get_time(), i=iteration, t1=process_time() - start, 
                          t2=score_time, s1=tr_score, s2=ts_score))
                if prev_score is not None and stop_score > prev_score:
                    print("{now} Stopping due to increase in score.".format(now=get_time()))
                prev_score = stop_score
            else:
                print("{now} Iteration {i} took {t1:.2f} seconds.".format(
                      now=get_time(), i=iteration, t1=process_time() - start))

            post_iter_func(iteration, self.s)

        if iteration == s["max_iter"]-1:
            print("{now} Stopped because reached max iteration.".format(now=get_time()))

        if s["verbose"] > 3:
            try:
                ax = plot_scores(tr_scores, ts_scores)
                ax.set_title(s["name"])
                plt.show()
            except Exception as e:
                print("{now} Failed to plot: {err}".format(now=get_time(), err=e))


    def _get_single_predictor(self, matrix, predict_func):
        """Construct a prediction vector by taking predictions at the
        items where the `matrix` parameter is non-zero. The actual values
        within the matrix are *not* used for predictions!
        """
        indices = np.where(matrix != 0)
        pred = np.zeros(len(indices[0]))
        for i in range(len(indices[0])):
            d = { "u": indices[0][i],
                  "i": indices[1][i]}
            pred[i] = predict_func(self.data, self.s, d)["prediction"]
        return pred


    def get_predictor(self, predict_func):
        # Training Predictor
        train_pred = self._get_single_predictor(self.train, predict_func)
        # Test Predictor
        test_pred = None
        if self.test is not None:
            test_pred = self._get_single_predictor(self.test, predict_func)

        return (train_pred, test_pred)


    def calc_score(self, true_data, predict_func):
        """ Calculate root MSE of the data predicted where the 
        `true_data` parameter (a matrix) is non-zero.
        """
        pred_data = self._get_single_predictor(true_data, predict_func)
        lin_true = true_data[true_data != 0]
        MSE = np.sum(np.square(pred_data - lin_true)) / len(pred_data)
        return np.sqrt(MSE)

data = {}
train = None

class ParallelSGD():
    default_settings = {
        "verbose": 5,
        "max_iter": 30,
        "calc_score": 3,
        "calc_train_score": True,
        "name": "SGD",
    }

    def __init__(self, settings, traind, test, init_data):
        self.s = copy.copy(SGD.default_settings)
        self.s.update(settings)
        global train
        train = traind
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
