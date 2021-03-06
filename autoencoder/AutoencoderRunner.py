import sys
import os

import numpy as np
import sklearn
import tensorflow as tf

import util
from MaskedDenoisingAutoencoder import DenoisingAutoencoder


""" Main file to run predictions with autoencoders.
Data preprocessing consists of normalizing the ratings within the -1, 1 range
which is needed since we use the tanh transfer function. 
This is similar to mean-centering the data.
The train_autoencoder function is responsible for training and making predictions.
It consists mainly of boilerplate code.
Parameters were chosen with a not very long run of Bayesian hyperparameter optimization.
The best results from the search were: 0.98592 (however the number of epochs was relatively low)

@author: gmeanti
"""


def prepare_data(X, mean_by=0):
    """Center around 0, mean center, and extract the missing value indices
    Params:
      X: input data matrix (nusers x nitems)
      mean_by: axis which contains the features 
               (either users: 0, or items: 1)
    Returns:
      (X_prepared, missing_indices)
    """

    # Extract missing value indices:
    maskedX = np.ma.array(X, mask=(X == 0), dtype="float32") # mask out the missing values

    # Values should be in [-1, 1] range (normally values are between 1 and 5)
    maskedX = (maskedX-3)/2

    # Mean center the values. Must be careful to only mean center the values
    # which are not in missing
    #maskedX = np.ma.apply_along_axis(lambda a: a - np.ma.mean(a),
    #                                                axis=mean_by,
    #                                                arr=maskedX)

    print("min: %f - mean: %f - std: %f - max: %f" % 
          (np.min(maskedX), np.mean(maskedX), np.std(maskedX), np.max(maskedX)))

    return (maskedX.data, maskedX.mask)


def untransform_data(X, mask):
    maskedX = np.ma.array(X, mask=mask) # mask out the missing values
    maskedX = maskedX*2+3
    return maskedX.data


def cross_val_score(k, sett, valid_size=0.1, data_axis=0):
    tr, _, _ = util.read_data(use_mask=True, test_percentage=0.0, dataset=dataset)
    
    flat_indices = np.where(tr != 0)
    data_size = len(flat_indices[0])
    fold_size = int(data_size * valid_size)

    ts_scores = []
    for i in range(k):
        test_indices = np.random.choice(range(data_size), fold_size, replace=False)
        test_2D_indices = (flat_indices[0][test_indices], flat_indices[1][test_indices])
        test_mask = np.zeros(tr.shape, dtype=bool)
        test_mask[test_2D_indices] = True # true if test

        test_fold = np.multiply(tr, test_mask)
        train_fold = np.multiply(tr, ~test_mask)
        tf.reset_default_graph()
        (cost, trerr, tserr) = train_autoencoder(sett, train_fold, test_fold, data_axis, make_predictions=False, dataset="CF")
        ts_scores.append(tserr)

    print("%s Mean %d-fold CV score: %f +- %f" % (util.get_time(), k, np.mean(ts_scores), np.std(ts_scores)))

# Item DAE: 0.986030
# User DAE: 0.992433

def train_autoencoder(sett, tr, ts, data_axis=0, make_predictions=False, dataset="CF"):
    """Train the denosing autoencoder for CF
    @param sett: dictionary of settings
    @param data_axis: whether to train a U-autoencoder (for users, data_axis=0)
            or I-autoencoder for items, data_axis=1
    @param make_predictions: whether to make predictions and write them to file or not.
            If not will use 10% of the training data to perform validation.
    """
    if data_axis == 0:
        train_size = sett["nusers"]
        feature_size = sett["nitems"]
    else:
        train_size = sett["nitems"]
        feature_size = sett["nusers"]

    tr, m_tr = prepare_data(tr, data_axis)
    ts, m_ts = prepare_data(ts, data_axis)
    if (data_axis == 1):
        tr = tr.T
        m_tr = m_tr.T
        ts = ts.T
        m_ts = m_ts.T
    print("%s Prepared data. Data axis is %d" % (util.get_time(), data_axis))


    batch = tf.Variable(0) # global step of the optimizer
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        sett["learning_rate"],       # Base learning rate.
        batch * sett["batch_size"],  # Current index into the dataset.
        train_size,          # Decay step.
        sett["learning_rate_decay"], # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate, collections=["autoencoder"])
    optimizer = tf.train.AdamOptimizer(learning_rate)

    model = DenoisingAutoencoder(name="autoencoder",
                                 n_input=feature_size,
                                 n_hidden=sett["hidden_size"],
                                 dropout_prob=sett["dropout_prob"],
                                 gaussian_prob=sett["gaussian_prob"],
                                 gaussian_std=sett["gaussian_std"],
                                 sap_prob=sett["sap_prob"],
                                 alpha_weight=sett["alpha"],
                                 beta_weight=sett["beta"],
                                 regl_weight=sett["regularization"],
                                 optimizer=optimizer,
                                 rseed=381328,
                                 batch=batch)
    model.init_saver([batch], os.path.join(sett["log_dir"], "model.ckpt"))

    batch_size = sett["batch_size"]
    train_indices = range(train_size)

    with tf.Session() as s:
        init = tf.global_variables_initializer()
        s.run(init)
        summary_writer = tf.summary.FileWriter(sett["log_dir"], graph=s.graph)

        for epoch in range(sett["num_epochs"]):
            print("%s Epoch %d" % (util.get_time(), epoch))
            # Randomize order of data samples at each epoch
            perm_indices = np.random.permutation(train_indices)
            # Index of data sample in this epoch
            run_index = 0

            for ibatch in range(train_size // batch_size):
                data_offset = (ibatch * batch_size) % (train_size - batch_size)
                batch_indices = perm_indices[data_offset:(data_offset+batch_size)]
                # Data for this batch
                batch_X = tr[batch_indices,:]
                batch_missing = m_tr[batch_indices,:]

                run_index += batch_size

                if run_index % sett["report_every"] == 0:
                    # print update and save summary for tensorboard
                    cost, trerr, tserr, summary = model.fit_summary(s, tr, m_tr, ts, m_ts)

                    print("%s step %d -- loss=%f -- train error=%f -- test error=%f" %
                        (util.get_time(), run_index, cost, trerr, tserr))

                    summary_writer.add_summary(summary, epoch*train_size + run_index)
                    summary_writer.flush()
                    sys.stdout.flush()
                else:
                    # Perform training
                    cost = model.fit(s, batch_X, batch_missing)

        # Make predictions and write them to file.
        if make_predictions:
            print("%s Making final predictions" % (util.get_time()))
            preds = model.predictions(s, tr)
            ts_pred = untransform_data(preds, m_ts)
            tr_pred = untransform_data(preds, m_tr)
            if data_axis == 1:
                ts_pred = ts_pred.T
                m_ts = m_ts.T
                tr_pred = tr_pred.T
                m_tr = m_tr.T
            util.write_predict(lambda u, i: ts_pred[u, i], np.invert(m_ts), sett["prediction_file"] + "_test.csv")
            util.write_predict(lambda u, i: tr_pred[u, i], np.invert(m_tr), sett["prediction_file"] + "_train.csv")
            print("%s Predictions written to %s" % (util.get_time(), sett["prediction_file"]))

        return (cost, trerr, tserr)


if __name__ == "__main__":
    print("%s Autoencoder Started" % (util.get_time()))
    dataset = "CF"

    # These settings give value (with axis 1): 0.990034 
    # (with axis 0): 0.985788
    settings = {
        "nusers" : 10000,
        "nitems" : 1000,
        "batch_size": 10,
        "num_epochs": 40,
        "report_every": 500,
        
        "hidden_size": 20,
        "learning_rate": 0.001,
        "learning_rate_decay": 0.9,
        "dropout_prob": 0.3,
        "gaussian_prob": 0.0,
        "gaussian_std": 0.08,
        "sap_prob": 0.01, # normally = 0.03-0.05
        "alpha": 1.2, # normally a=0.9,b=0.2;a=1,b=0.1 works better(ts=0.982);a=1.2,ts=0.981;
        "beta": 0.1,  # beta << alpha otherwise a lot of overfitting occurs
        "regularization": 0.1,

        "log_dir": "./saved_data/logs/single_layer",
        "prediction_file": "./saved_data/autoencoder_predict.csv",
    }

    # if dataset == "CF":
        # cross_val_score(3, settings, 0.1, data_axis=0)
        # cross_val_score(3, settings, 0.1, data_axis=1)
    # elif dataset == "movielens":
        # settings["nusers"] = 6040
        # settings["nitems"] = 3952
        # train_autoencoder(settings, data_axis=0, make_predictions=False, dataset="movielens")

    ## This creates 2 predictors
    tr, _, ts = util.read_data(use_mask=True, dataset=dataset)
    print("%s Read data from dataset %s" % (util.get_time(), dataset))
    
    settings["prediction_file"] = "./saved_data/autoenc_ax0_pred"
    train_autoencoder(settings, tr, ts, data_axis=0, make_predictions=True, dataset="CF")

    tf.reset_default_graph()
    settings["hidden_size"] = 60
    settings["prediction_file"] = "./saved_data/autoenc_ax1_pred"
    train_autoencoder(settings, tr, ts, data_axis=1, make_predictions=True, dataset="CF")
