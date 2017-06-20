from __future__ import print_function, division, absolute_import
import traceback

import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization

import util
import AutoencoderRunner

SAVE_PATH = "../saved_data/bayes_opt/results.csv"

mut_settings = {
    "hidden_size": [10, 100],
    "learning_rate": [0.0005, 0.1],
    "learning_rate_decay": [0.9, 0.99],
    "dropout_prob": [0.0, 0.5],
    "gaussian_prob": [0.0, 0.2],
    "gaussian_std": [0.01, 1.0],
    "sap_prob": [0.0, 0.1], # normally = 0.03-0.05
    "alpha": [0.1, 2.0], # normally a=0.9,b=0.2;a=1,b=0.1 works better(ts=0.982);a=1.2,ts=0.981;
    "beta": [0.1, 2.0],  # beta << alpha otherwise a lot of overfitting occurs
    "regularization": [0.1, 100.0],
}

gp_params = {"alpha": 1e-4}


def target(hidden_size, learning_rate, learning_rate_decay, dropout_prob, 
           gaussian_prob, gaussian_std, sap_prob, alpha, beta, regularization):
    fixed_settings = {
        "nusers" : 10000,
        "nitems" : 1000,
        "batch_size": 10,
        "num_epochs": 20,
        "report_every": 500,
        
        "log_dir": "./saved_data/logs/single_layer",
        "prediction_file": "./saved_data/autoencoder_predict.csv",
    }

    fixed_settings["hidden_size"] = int(hidden_size)
    fixed_settings["learning_rate"] = learning_rate
    fixed_settings["learning_rate_decay"] = learning_rate_decay
    fixed_settings["dropout_prob"] = dropout_prob
    fixed_settings["gaussian_prob"] = gaussian_prob
    fixed_settings["gaussian_std"] = gaussian_std
    fixed_settings["sap_prob"] = sap_prob
    fixed_settings["alpha"] = alpha
    fixed_settings["beta"] = beta
    fixed_settings["regularization"] = regularization

    tf.reset_default_graph()
    cost, trerr, tserr = AutoencoderRunner.train_autoencoder(fixed_settings, data_axis=0, make_predictions=False, dataset="CF")

    # Need to negate error since BayesOptimizization tries to maximize function
    return -tserr


def run_optimization():
    # Define the optimizer object
    autoencBO = BayesianOptimization(target, mut_settings)

    max_iter = 1000
    #report_every = 5
    init_points = 30

    print("%s Starting optimization with %d initial points, %d iterations." % (util.get_time(), init_points, max_iter))
    # Higher kappa -> higher exploration (lower exploitation)
    autoencBO.maximize(init_points=init_points, n_iter=max_iter, acq="ucb", kappa=8, **gp_params)

    # Save resulting data
    autoencBO.points_to_csv(SAVE_PATH)
    print("%s Results written to %s" % (util.get_time(), SAVE_PATH))


if __name__ == "__main__":
    print("%s Starting Bayesian optimizer." % (util.get_time()))

    try:
        run_optimization()
    except Exception as e:
        print("%s Encountered exception %s" % (util.get_time(), e))
        print(traceback.format_exc())

    print("%s Optimization finished." % (util.get_time()))
