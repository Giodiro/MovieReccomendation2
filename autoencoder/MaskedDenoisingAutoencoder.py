from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np



class DenoisingAutoencoder:
    """Implementation of a denoising autoencoder for collaborative filtering
    This class is quite generic in that it implements 3 types of noise
     - Gaussian noise
     - Masking noise
     - Salt and Pepper noise
    which can be applied to the input in different proportions.
    As already done in https://arxiv.org/pdf/1603.00806.pdf
    we deal with the sparsity of the input matrix typical of the CF problem
    by ignoring (i.e. setting to zero) the absent values, so that they do not concur
    in the gradient propagation procedure.
    However no optimizations taking advantage of the input sparsity have been applied (TODO)

    Transfer function: tanh
    Layers: 1 hidden layer of size `n_hidden`
    """
    def __init__(self, name, n_input, n_hidden, dropout_prob,
                 gaussian_prob, gaussian_std, sap_prob,
                 alpha_weight, beta_weight, regl_weight,
                 batch, optimizer):
        """Initialize the weights and the graph for the autoencoder
        @param name: identifier of this neural network used for grouping the 
                emitted summaries
        @param n_input: size of each input vector (feature dimensionality)
        @param n_hidden: size of the hidden layer (should be much smaller than `n_input`)
        @param dropout_prob: probability with which to apply masking noise
        @param gaussian_prob: probability with which to apply gaussian noise
        @param gaussian_std: standard deviation of gaussian noise
        @param sap_prob: probability with which to apply salt and pepper noise
        @param alpha_weight: alpha is a weight for the loss of dropped out ratings.
                Increasing alpha will make the network focus more on reconstruction
        @param beta_weight: beta is counterpart to alpha: weighs the loss of 
                non-dropped out ratings.
        @param regl_weight: regularization applied to the weight matrix
        @param batch: tensorflow variable used to indicate the current data batch
        @param optimizer: the tensorflow optimizer to use for backpropagation
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.name = name

        self.weights = self._init_weights()

        """
        In the input there will be many 0 values (missing) and some actual values.
        We don't care about the 0 values, but want to perform dropout on the actual 
        values. The loss function differs depending on whether we are talking about
        dropped out values (weighted with alpha) or kept values (weighted with beta),
        but it does not take into consideration the missing values.
        See https://arxiv.org/pdf/1603.00806.pdf for more details
        """

        # Input data
        self.input         = tf.placeholder(tf.float32, [None, self.n_input], name="input")
        # Boolean mask selecting the invalid (missing) values in the input data
        self.input_mask    = tf.placeholder(tf.bool, [None, self.n_input], name="input_mask") # True when input is invalid

        # Store the parameters in tensorflow constants.
        # dropout probability is converted to keep probability to simplify calculations below
        self.keep_prob     = tf.constant(1-dropout_prob, dtype=tf.float32, shape=[], name="keep_probability")
        self.gaussian_prob = tf.constant(gaussian_prob, dtype=tf.float32, shape=[], name="gaussian_noise_probability")
        self.sap_prob      = tf.constant(sap_prob, dtype=tf.float32, shape=[], name="SAP_probability")
        self.alpha         = tf.constant(alpha_weight, dtype=tf.float32, shape=[], name="alpha")
        self.beta          = tf.constant(beta_weight, dtype=tf.float32, shape=[], name="beta")
        self.regl          = tf.constant(regl_weight, dtype=tf.float32, shape=[], name="regularization")

        zero = tf.constant(0, dtype=tf.float32)
        one = tf.constant(1, dtype=tf.float32)

        input_shape = self.input.get_shape()
        input_type = self.input.dtype
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = self.keep_prob
        random_tensor += tf.random_uniform(tf.shape(self.input),
                                           seed=None,
                                           dtype=input_type)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)

        with tf.name_scope("noise"):
            # Apply noise to the input.
            self.noisy_input = self.add_gaussian_noise(self.input,
                                                       self.gaussian_prob,
                                                       gaussian_std)
            self.noisy_input = self.add_salt_and_pepper_noise(self.noisy_input,
                                                              self.sap_prob)
            
            self.noisy_input = tf.div(self.noisy_input, self.keep_prob) * binary_tensor
            self.noisy_input.set_shape(input_shape)

        with tf.name_scope("masks"):
            # True if a rating is valid and it has been dropped out
            self.alpha_mask = tf.logical_and(tf.equal(binary_tensor, zero),
                                             tf.logical_not(self.input_mask))
            self.alpha_mask.set_shape(input_shape)
            # True if a rating is valid and it has not been dropped out
            self.beta_mask = tf.logical_and(tf.equal(binary_tensor, one),
                                            tf.logical_not(self.input_mask))
            self.beta_mask.set_shape(input_shape)
            
            tf.summary.scalar("alpha_mask_size",
                tf.reduce_sum(tf.cast(self.alpha_mask, tf.int32)),
                collections=[self.name])
            tf.summary.scalar("beta_mask_size",
                tf.reduce_sum(tf.cast(self.beta_mask, tf.int32)),
                collections=[self.name])
        
        with tf.name_scope("activations"):
            # Encoder
            #self.noisy_input = tf.tanh(self.noisy_input)
            self.hidden = tf.nn.tanh(tf.add(
                            tf.matmul(self.noisy_input, self.weights['W1']),
                            self.weights['b1']))
            # Decoder
            self.reconstruction = tf.nn.tanh(tf.add(
                            tf.matmul(self.hidden, self.weights['W2']), 
                            self.weights['b2']))
            tf.summary.histogram('reconstruction', 
                tf.boolean_mask(self.reconstruction, 
                                tf.logical_not(self.input_mask)), collections=[self.name])
            tf.summary.histogram('hidden', self.hidden, collections=[self.name])
            tf.summary.histogram('input',
                tf.boolean_mask(self.input,
                                tf.logical_not(self.input_mask)), collections=[self.name])
            #print("Reconstruction shape: %s" % (self.reconstruction.get_shape().as_list()))

        with tf.name_scope("loss"):
            # Calculate loss
            self.alpha_cost = tf.reduce_sum(tf.squared_difference(
                                    tf.boolean_mask(self.reconstruction, self.alpha_mask), 
                                    tf.boolean_mask(self.input, self.alpha_mask))) * self.alpha
            self.beta_cost  = tf.reduce_sum(tf.squared_difference(
                                    tf.boolean_mask(self.reconstruction, self.beta_mask),
                                    tf.boolean_mask(self.input, self.beta_mask))) * self.beta
            # Regularization
            self.regl_cost = self.regl * tf.add(tf.nn.l2_loss(self.weights['W1']),
                                                tf.nn.l2_loss(self.weights['W2']))
            self.cost = self.alpha_cost + self.beta_cost + self.regl_cost
 
            tf.summary.scalar('alpha_cost', self.alpha_cost, collections=[self.name])
            tf.summary.scalar('beta_cost', self.beta_cost, collections=[self.name])
            tf.summary.scalar('regl_cost', self.regl_cost, collections=[self.name])
            tf.summary.scalar('cost', self.cost, collections=[self.name])

        with tf.name_scope("accuracy"):
            self.train_err = self.rmse(self.reconstruction, self.input, self.input_mask, "train_predictions")
            tf.summary.scalar("train_error", self.train_err, collections=[self.name])

            self.test_input = tf.placeholder(tf.float32, [None, self.n_input],
                    name="test_input")
            self.test_input_mask = tf.placeholder(tf.bool, [None, self.n_input], 
                    name="test_input_mask") # True when input is invalid
            self.test_err = self.rmse(self.reconstruction, self.test_input, self.test_input_mask, "test_predictions")
            tf.summary.scalar("test_error", self.test_err, collections=[self.name])

        with tf.name_scope("optimizer"):
            self.optimizer = optimizer.minimize(self.cost, global_step=batch)

        self.summary_op = tf.summary.merge_all(key=self.name)


    def add_gaussian_noise(self, X, noise_prob, std):
        if noise_prob == 0.0:
            return X
        random_tensor = noise_prob
        random_tensor += tf.random_uniform(shape=tf.shape(X), seed=None, dtype=X.dtype)
        binary_tensor = tf.floor(random_tensor)
        noise_tensor = tf.random_normal(shape=tf.shape(X), mean=0.0, stddev=std, dtype=X.dtype) * \
                        binary_tensor
        return X + noise_tensor

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


    def _init_weights(self):
        weights = {}
        with tf.name_scope("hidden_layer"):
            weights["W1"] = tf.get_variable('W1', 
                shape=[self.n_input, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            self.variable_summaries(weights["W1"], "weights")

            weights["b1"] = tf.get_variable('b1', 
                shape=[self.n_hidden], 
                initializer=tf.zeros_initializer())
            self.variable_summaries(weights["b1"], "biases")
        with tf.name_scope("reconstruction_layer"):
            weights["W2"] = tf.get_variable('W2', 
                shape=[self.n_hidden, self.n_input],
                initializer=tf.contrib.layers.xavier_initializer())
            self.variable_summaries(weights["W2"], "weights")

            weights["b2"] = tf.get_variable('b2', 
                shape=[self.n_input], 
                initializer=tf.zeros_initializer())
            self.variable_summaries(weights["b2"], "biases")

        return weights


    def variable_summaries(self, var, name):
      """Attach a lot of summaries to a Tensor."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean, collections=[self.name])
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev, collections=[self.name])
        tf.summary.scalar('max/' + name, tf.reduce_max(var), collections=[self.name])
        tf.summary.scalar('min/' + name, tf.reduce_min(var), collections=[self.name])
        tf.summary.histogram(name, var, collections=[self.name])


    def partial_fit(self, sess, input, input_mask):
        fd = {
            self.input: input,
            self.input_mask: input_mask,
        }
        cost, opt, predictions = sess.run((self.cost, self.optimizer, self.reconstruction),
                             feed_dict = fd)
        return cost, predictions


    def partial_fit_summary(self, sess, input, input_mask, test, test_mask):
        fd = {
            self.input: input,
            self.input_mask: input_mask,
            self.test_input: test,
            self.test_input_mask: test_mask,
        }

        cost, predictions, trerr, tserr, summary = sess.run(
            (self.cost, self.reconstruction, self.train_err, self.test_err,
             self.summary_op), feed_dict = fd)
        return cost, predictions, trerr, tserr, summary


    def predictions(self, sess, input):
        fd = {
            self.input: input,
            self.keep_prob: 1.0,
            self.noise_prob: 0.0,
            self.sap_prob: 0.0,
        }
        predictions = sess.run((self.reconstruction), feed_dict=fd)
        return predictions


    def hidden_activations(self, sess, input):
        fd = {
            self.input: input,
            self.keep_prob: self.dropout_probability,
            self.noise_prob: self.gaussian_prob,
            self.sap_prob: self.salt_and_pepper_prob,
        }
        hid = sess.run((self.hidden), feed_dict=fd)
        return hid


    def init_saver(self, more_vars, save_file):
        var_list = list(self.weights.values()) + more_vars
        self.saver = tf.train.Saver(var_list=var_list)
        self.saver_filename = save_file


    def save_model(self, sess, global_step=None):
        return self.saver.save(sess, self.saver_filename, global_step=global_step)


    def restore_model(self, sess):
        return self.saver.restore(sess, self.saver_filename)


    def rmse(self, preds, truth, mask, name):
        """RMSE"""
        notmask = tf.logical_not(mask)
        valid_preds = tf.boolean_mask(preds, notmask)
        valid_preds = valid_preds*2 + 3
        tf.summary.histogram(name, valid_preds, collections=[self.name])

        valid_truth = tf.boolean_mask(truth, notmask)
        valid_truth = valid_truth*2 + 3
        tf.summary.histogram(name + "_truth", valid_truth, collections=[self.name])
        err = tf.sqrt(tf.reduce_mean(tf.squared_difference(valid_truth, valid_preds)))
        return err
