import util
from MaskedDenoisingAutoencoder import DenoisingAutoencoder

import numpy as np
import sklearn
import tensorflow as tf

import sys
import os

def getLayerName(hl):
    return "hl_%d" % hl

class StackedDenoisingAutoencoder:
    def __init__(self, n_input, n_hidden_layers, dropout_prob,
                 sap_prob,
                 alpha_weight, beta_weight, regl_weight,
                 batch, optimizer):
        self.n_input = n_input
        self.dropout_probability = dropout_prob
        self.salt_and_pepper_prob = sap_prob
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight
        self.regl_weight = regl_weight
        self.n_hidden_layers = n_hidden_layers

        self.keep_prob = tf.placeholder(tf.float32, [], name="keep_probability")
        self.sap_prob = tf.placeholder(tf.float32, [], name="SAP_probability")

        """
        In the input there will be many 0 values (missing) and some actual values.
        We don't care about the 0 values, but want to perform dropout on the actual values.
        Then the loss function differs depending on whether we are talking about dropped out values or kept values,
        but it does not take into consideration the missing values.
        """

        self.input = tf.placeholder(tf.float32, [None, self.n_input], name="input")
        self.input_mask = tf.placeholder(tf.bool, [None, self.n_input], name="input_mask") # True when input is invalid
        self.alpha = tf.placeholder(tf.float32, [], name="alpha")
        self.beta = tf.placeholder(tf.float32, [], name="beta")
        self.regl = tf.placeholder(tf.float32, [], name="regularization")
        self.weights = self._init_weights()

        # Now we define the refining network:
        binary_tensor = self._apply_masking(self.input, self.keep_prob)

        zero = tf.constant(0, dtype=tf.float32)
        one = tf.constant(1, dtype=tf.float32)
        # True if a rating is valid and it has been dropped out
        self.alpha_mask = tf.logical_and(tf.equal(binary_tensor, zero),
                                         tf.logical_not(self.input_mask))
        self.alpha_mask.set_shape(self.input.get_shape())
        # True if a rating is valid and it has not been dropped out
        self.beta_mask = tf.logical_and(tf.equal(binary_tensor, one),
                                        tf.logical_not(self.input_mask))
        self.beta_mask.set_shape(self.input.get_shape())

        # Apply noise to input
        dropout_input = tf.div(self.input, self.keep_prob) * binary_tensor
        dropout_input.set_shape(self.input.get_shape())
        dropout_input = self.add_salt_and_pepper_noise(dropout_input,
                                                       self.sap_prob)


        self.hidden = [None for i in range(self.n_hidden_layers)]
        self.reconstruction = [None for i in range(self.n_hidden_layers)]
        for hl in range(self.n_hidden_layers):
            if hl == 0:
                hid_in = self.input
            else:
                hid_in = self.hidden[hl-1]
            self.hidden[hl] = tf.tanh(tf.nn.bias_add(
                            tf.matmul(hid_in, self.weights[hl]["W1"]),
                            self.weights[hl]["b1"]))

        for hl in range(self.n_hidden_layers):
            rev_hl = list(reversed(range(self.n_hidden_layers)))[hl]
            if hl == 0:
                rec_in = self.hidden[-1] # Last hidden output is first input for reconstruction
            else:
                rec_in = self.reconstruction[hl-1]
            self.reconstruction[hl] = tf.tanh(tf.nn.bias_add(
                            tf.matmul(rec_in, self.weights[rev_hl]["W2"]), 
                            self.weights[rev_hl]["b2"]))

        tf.summary.histogram('reconstruction', 
            tf.boolean_mask(self.reconstruction[-1], 
                            tf.logical_or(self.beta_mask, self.alpha_mask)))
        tf.summary.histogram('input', 
            tf.boolean_mask(self.input,
                            tf.logical_or(self.beta_mask, self.alpha_mask)))

        # Calculate loss
        last_rec = self.reconstruction[-1]
        alpha_cost = tf.subtract(tf.boolean_mask(last_rec, self.alpha_mask), 
                                 tf.boolean_mask(self.input, self.alpha_mask))
        beta_cost  = tf.subtract(tf.boolean_mask(last_rec, self.beta_mask),
                                 tf.boolean_mask(self.input, self.beta_mask))
        self.cost = tf.add(tf.multiply(self.alpha, tf.reduce_sum(tf.pow(alpha_cost, 2.0))),
                      tf.multiply(self.beta , tf.reduce_sum(tf.pow(beta_cost , 2.0))))

        # Regularization
        for hl in range(self.n_hidden_layers):
            self.cost += self.regl * (tf.nn.l2_loss(self.weights[hl]["W1"]) +
                                      tf.nn.l2_loss(self.weights[hl]["W2"]))

        tf.summary.scalar('loss', self.cost)

        self.summary_op = tf.summary.merge_all()

        self.optimizer = optimizer.minimize(self.cost, global_step=batch)

    def partial_fit(self, sess, input, input_mask):
        fd = {
            self.input: input,
            self.input_mask: input_mask,
            self.keep_prob: self.dropout_probability,
            self.sap_prob: self.salt_and_pepper_prob,
            self.alpha: self.alpha_weight,
            self.beta: self.beta_weight,
            self.regl: self.regl_weight,
        }
        cost, opt, predictions = sess.run((self.cost, self.optimizer, self.reconstruction[-1]),
                             feed_dict = fd)
        return cost, predictions

    def partial_fit_summary(self, sess, input, input_mask):
        fd = {
            self.input: input,
            self.input_mask: input_mask,
            self.keep_prob: self.dropout_probability,
            self.sap_prob: self.salt_and_pepper_prob,
            self.alpha: self.alpha_weight,
            self.beta: self.beta_weight,
            self.regl: self.regl_weight,
        }
        cost, opt, predictions, summary = sess.run(
            (self.cost, self.optimizer, self.reconstruction[-1], 
             self.summary_op), feed_dict = fd)
        return cost, predictions, summary

    def _apply_masking(self, X, keep_prob):
        noise_shape = tf.shape(X)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape,
                                           seed=None,
                                           dtype=X.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        return binary_tensor

    def _init_weights(self):
        weights = []
        for hl in range(self.n_hidden_layers):
            weights.append({})
            ln = getLayerName(hl)
            with tf.name_scope(ln):
                with tf.variable_scope(ln, reuse=True):
                    weights[hl]["W1"] = tf.get_variable('W1')
                    weights[hl]["b1"] = tf.get_variable('b1')
                    weights[hl]["W2"] = tf.get_variable('W2')
                    weights[hl]["b2"] = tf.get_variable('b2')
        return weights

    def test_predictions(self, sess, input):
        fd = {
            self.input: input,
            self.keep_prob: 1.0,
            self.sap_prob: 0.0,
        }
        predictions = sess.run((self.reconstruction[-1]), feed_dict=fd)
        return predictions

    def init_saver(self, more_vars, save_file):
        weight_list = []
        for wdict in self.weights: weight_list.extend(list(wdict.values())) 
        var_list = weight_list + more_vars
        self.saver = tf.train.Saver(var_list=var_list)
        self.saver_filename = save_file


    def add_salt_and_pepper_noise(self, X, noise_prob):
        if noise_prob == 0.0:
            return X
        mn = tf.reduce_min(X)
        mx = tf.reduce_max(X)

        half_noise_prob = 1 - noise_prob/2.0

        min_tensor = tf.floor(half_noise_prob + 
                            tf.random_uniform(shape=tf.shape(X), seed=None, dtype=X.dtype))
        min_indices = tf.to_int32(
            tf.where(tf.equal(min_tensor, tf.constant(1, dtype=X.dtype))))
        num_true = tf.shape(min_indices)[0]

        X = tf.add(tf.multiply(X, min_tensor),
                   tf.scatter_nd(min_indices, tf.fill([num_true], mn), shape=tf.shape(X)))

        max_tensor = tf.floor(half_noise_prob + 
                            tf.random_uniform(shape=tf.shape(X), seed=None, dtype=X.dtype))
        max_indices = tf.to_int32(
            tf.where(tf.equal(max_tensor, tf.constant(1, dtype=X.dtype))))
        num_true = tf.shape(max_indices)[0]

        X = tf.add(tf.multiply(X, max_tensor),
                   tf.scatter_nd(max_indices, tf.fill([num_true], mx), shape=tf.shape(X)))

        return X


    def save_model(self, sess, global_step=None):
        return self.saver.save(sess, self.saver_filename, global_step=global_step)

    def restore_model(self, sess):
        return self.saver.restore(sess, self.saver_filename)



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

def error(preds, truth, mask):
    """RMSE"""
    preds = np.ma.array(untransform_data(preds, mask), mask=mask)
    truth = np.ma.array(untransform_data(truth, mask), mask=mask)

    return np.sqrt(np.ma.mean(np.ma.power(preds - truth, 2)))


def main_stacked():
    sett = {
        "nusers" : 10000,
        "nitems" : 1000,
        "batch_size": 10,
        "num_epochs": 50,
        "report_every": 500,
        
        "n_hidden_layers": 2,
        "hidden_sizes": [50, 20],
        "hl_0" : {
            "restore": False,
            "learning_rate": 0.03,
            "learning_rate_decay": 0.9,
            "dropout_prob": 0.85,
            "gaussian_prob": 0,
            "gaussian_std": 0.08,
            "sap_prob": 0.05,
            "alpha": 2,
            "beta": 0.5,
            "regularization": 0.03,
            "log_dir": "./saved_data/logs/hidden_0",
            "num_epochs": 80,
        },
        "hl_1" : {
            "restore": False,
            "learning_rate": 0.0001,
            "learning_rate_decay": 0.9,
            "dropout_prob": 1,
            "gaussian_prob": 0.8,
            "gaussian_std": 0.02,
            "sap_prob": 0.1,
            "alpha": 1,
            "beta": 1,
            "regularization": 0.2,
            "log_dir": "./saved_data/logs/hidden_1",
            "num_epochs": 80,
        },
        "learning_rate": 0.004,
        "dropout_prob": 0.85,
        "sap_prob": 0.05,
        "alpha": 1.2,
        "beta": 0.8,
        "regularization": 0.03,
        "learning_rate_decay": 0.8,

        "restore": False,
        "log_dir": "./saved_data/logs/refinement",

        "prediction_file": "./saved_data/autoencoder_predict.csv",
    }

    data_axis = 0
    do_test = True
    if do_test:
        tr, ts, msk = util.read_data(use_mask=False, test_percentage=0.1)
    else:
        tr, _, ts = util.read_data(use_mask=True)

    print("%s read data." % (util.get_time()))
    tr, m_tr = prepare_data(tr, data_axis)
    ts, m_ts = prepare_data(ts, data_axis)
    if (data_axis == 1):
        tr = tr.T
        m_tr = m_tr.T
        ts = ts.T
        m_ts = m_ts.T
    print("%s prepared data." % util.get_time())

    n_hidden_layers = sett["n_hidden_layers"]
    hidden_sizes = sett["hidden_sizes"]
    train_size = tr.shape[0] # This is number of examples

    hidden_models = []
    # Create hidden layer autoencoders
    for hl in range(n_hidden_layers):
        ln = getLayerName(hl)
        with tf.variable_scope(ln):
            with tf.name_scope(ln):
                if hl == 0:
                    input_size = tr.shape[1]  # This is feature size
                else:
                    input_size = hidden_sizes[hl-1]

                batch = tf.Variable(0) # global step of the optimizer
                # Decay once per epoch, using an exponential schedule starting at 0.01.
                learning_rate = tf.train.exponential_decay(
                    sett[ln]["learning_rate"],       # Base learning rate.
                    batch * sett["batch_size"],  # Current index into the dataset.
                    train_size,                    # Decay step.
                    sett[ln]["learning_rate_decay"], # Decay rate.
                    staircase=True)
                tf.summary.scalar('learning_rate', learning_rate, collections=[ln])
                optimizer = tf.train.AdamOptimizer(learning_rate)
                model = MaskingNoiseAutoencoder(name=ln,
                                                n_input=input_size,
                                                n_hidden=hidden_sizes[hl],
                                                dropout_prob=sett[ln]["dropout_prob"],
                                                gaussian_prob=sett[ln]["gaussian_prob"],
                                                gaussian_std=sett[ln]["gaussian_std"],
                                                sap_prob=sett[ln]["sap_prob"],
                                                alpha_weight=sett[ln]["alpha"],
                                                beta_weight=sett[ln]["beta"],
                                                regl_weight=sett[ln]["regularization"],
                                                optimizer=optimizer,
                                                batch=batch)
                model.init_saver([batch], os.path.join(sett[ln]["log_dir"], "model.ckpt"))
                hidden_models.append(model)

    # Create refinemenet network
    #with tf.variable_scope("refinement"):
    batch = tf.Variable(0) # global step of the optimizer
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        sett["learning_rate"],       # Base learning rate.
        batch * sett["batch_size"],  # Current index into the dataset.
        train_size,                    # Decay step.
        sett["learning_rate_decay"], # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    ref_model = StackedDenoisingAutoencoder(n_input=tr.shape[1], 
                                            n_hidden_layers=n_hidden_layers,
                                            dropout_prob=sett["dropout_prob"],
                                            sap_prob=sett["sap_prob"],
                                            alpha_weight=sett["alpha"],
                                            beta_weight=sett["beta"],
                                            regl_weight=sett["regularization"],
                                            batch=batch,
                                            optimizer=optimizer)
    ref_model.init_saver([batch], os.path.join(sett["log_dir"], "model.ckpt"))

    with tf.Session() as s:
        init = tf.global_variables_initializer()
        s.run(init)

        train_indices = range(train_size)

        hidden_activs = []
        # Pretraining
        for hl in range(n_hidden_layers):
            ln = getLayerName(hl)
            model = hidden_models[hl]
            if sett[ln]["restore"]:
                model.restore_model(s)
                print("%s Restored model for layer %d" % (util.get_time(), hl))
                continue
            print("%s Starting pretraining layer %d" % (util.get_time(), hl))
            with tf.variable_scope(ln, reuse=True):
                with tf.name_scope(ln):
                    if hl == 0:
                        train_data = tr
                        train_mask = m_tr
                    else:
                        train_data = hidden_activs[hl-1]
                        train_mask = np.zeros_like(train_data) # no masking
                    summary_writer = tf.summary.FileWriter(sett[ln]["log_dir"], graph=s.graph)
                    for epoch in range(sett[ln]["num_epochs"]):
                        perm_indices = np.random.permutation(train_indices)
                        for ibatch in range(train_size // sett["batch_size"]):
                            data_offset = (ibatch * sett["batch_size"]) % (train_size - sett["batch_size"])
                            batch_indices = perm_indices[data_offset:(data_offset+sett["batch_size"])]
                            batch_data = train_data.take(batch_indices, 0) ## TODO: change this to normal indexing
                            batch_missing = train_mask.take(batch_indices, 0)

                            step = ibatch * sett["batch_size"]
                            if step % sett["report_every"] == 0:
                                cost, predictions, summary = model.partial_fit_summary(s, batch_data, batch_missing)
                                summary_writer.add_summary(summary, step * (epoch+1))
                                summary_writer.flush()
                                print("%s step %d, epoch %d -- loss=%f" % (
                                    util.get_time(), step, epoch, cost))
                            else:
                                cost, predictions = model.partial_fit(s, batch_data, batch_missing)
                    # Save this model
                    save_path = model.save_model(s)
                    print("%s Model saved in file %s" % (util.get_time(), save_path))

                    # Take hidden layer output to input to the next hidden layer
                    hidden_activs.append(model.hidden_activations(s, train_data))

        # Refine network
        summary_writer = tf.summary.FileWriter(sett["log_dir"], graph=s.graph)
        train_data = tr
        train_mask = m_tr

        if sett["restore"]:
            ref_model.restore_model(s)
            print("%s Restored refinement model" % (util.get_time()))
        else:
            print("%s Starting refinement learning" % (util.get_time()))
            for epoch in range(sett["num_epochs"]):
                perm_indices = np.random.permutation(train_indices)
                for ibatch in range(train_size // sett["batch_size"]):
                    data_offset = (ibatch * sett["batch_size"]) % (train_size - sett["batch_size"])
                    batch_indices = perm_indices[data_offset:(data_offset+sett["batch_size"])]
                    batch_data = train_data.take(batch_indices, 0) ## TODO: change this to normal indexing
                    batch_missing = train_mask.take(batch_indices, 0)
                    
                    step = ibatch * sett["batch_size"]
                    if step % sett["report_every"] == 0:
                        cost, predictions, summary = ref_model.partial_fit_summary(s, batch_data, batch_missing)
                        summary_writer.add_summary(summary, step * (epoch+1))
                        summary_writer.flush()
                        ts_err = np.NaN
                        if do_test:
                            ts_pred = ref_model.test_predictions(s, tr)
                            ts_err = error(ts_pred, ts, m_ts)

                        print("%s step %d, epoch %d -- loss=%f -- train error=%f -- test error=%f" % (
                            util.get_time(), step, epoch, cost, 
                            error(predictions, batch_data, batch_missing),
                            ts_err))
                    else:
                        cost, predictions = ref_model.partial_fit(s, batch_data, batch_missing)

            # Save this model
            save_path = ref_model.save_model(s)
            print("%s Model saved in file %s" % (util.get_time(), save_path))


def main(sett, data_axis=0, make_predictions=False):
    if data_axis == 0:
        train_size = sett["nusers"]
        feature_size = sett["nitems"]
    else:
        train_size = sett["nitems"]
        feature_size = sett["nusers"]

    if make_predictions:
        tr, _, ts = util.read_data(use_mask=True)
    else:
        tr, ts, msk = util.read_data(use_mask=False, test_percentage=0.1)

    print("%s read data." % (util.get_time()))
    tr, m_tr = prepare_data(tr, data_axis)
    ts, m_ts = prepare_data(ts, data_axis)
    if (data_axis == 1):
        tr = tr.T
        m_tr = m_tr.T
        ts = ts.T
        m_ts = m_ts.T
    print("%s prepared data." % util.get_time())


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
                                 batch=batch)

    batch_size = sett["batch_size"]

    with tf.Session() as s:
        init = tf.global_variables_initializer()
        s.run(init)

        summary_writer = tf.summary.FileWriter(sett["log_dir"], graph=s.graph)

        train_indices = range(train_size)

        for epoch in range(sett["num_epochs"]):
            print("Epoch %d" % epoch)
            perm_indices = np.random.permutation(train_indices)
            run_index = 0

            for ibatch in range(train_size // batch_size):
                data_offset = (ibatch * batch_size) % (train_size - batch_size)
                batch_indices = perm_indices[data_offset:(data_offset+batch_size)]
                # Data for this batch
                batch_X = tr.take(batch_indices, 0) ## TODO: change this to normal indexing
                batch_missing = m_tr.take(batch_indices, 0)

                run_index += batch_size

                if run_index % sett["report_every"] == 0:
                    # Output summary
                    cost, predictions, trerr, tserr, summary = \
                        model.partial_fit_summary(s, tr, m_tr, ts, m_ts)

                    print("%s step %d -- loss=%f -- train error=%f -- test error=%f" %
                        (util.get_time(), run_index, cost, trerr, tserr))

                    summary_writer.add_summary(summary, epoch*train_size + run_index)
                    summary_writer.flush()
                    sys.stdout.flush()
                else:
                    # Perform training
                    cost, predictions = model.partial_fit(s, batch_X, batch_missing)

        # Make predictions
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

if __name__ == "__main__":
    print("%s tf_sgd started" % (util.get_time()))
    settings = {
        "nusers" : 10000,
        "nitems" : 1000,
        "batch_size": 10,
        "num_epochs": 30,
        "report_every": 500,
        
        "hidden_size": 20,
        "learning_rate": 0.004,
        "learning_rate_decay": 0.9,
        "dropout_prob": 0.8,
        "gaussian_prob": 0,
        "gaussian_std": 0.08,
        "sap_prob": 0.01, # normally = 0.03-0.05
        "alpha": 1., # normally a=0.9,b=0.2;a=1,b=0.1 works better(ts=0.982);a=1.2,ts=0.981;
        "beta": 0.1,  # beta << alpha otherwise a lot of overfitting occurs
        "regularization": 0.1,

        "log_dir": "./saved_data/logs/single_layer",
        "prediction_file": "./saved_data/autoencoder_predict.csv",
    }

    settings["prediction_file"] = "./saved_data/autoenc_ax1_pred"
    main(settings, data_axis=1, make_predictions=True)
    
    settings["prediction_file"] = "./saved_data/autoenc_ax0_pred"
    main(settings, data_axis=0, make_predictions=True)

