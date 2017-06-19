from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

"""File only contains the class for the denoising autoencoder.

@author: gmeanti
"""

class DenoisingAutoencoder:
    """Implementation of a denoising autoencoder for collaborative filtering
    This class is quite generic in that it implements 3 types of noise
     - Gaussian noise
     - Masking noise (particularly important for the CF problem)
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
                 batch, rseed, optimizer):
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
        @param rseed: random seed for various noise sources
        @param optimizer: the tensorflow optimizer to use for backpropagation
        """
        self.n_input           = n_input
        self.n_hidden          = n_hidden
        self.name              = name
        self.keep_prob_val     = 1 - dropout_prob
        self.gaussian_prob_val = gaussian_prob
        self.sap_prob_val      = sap_prob

        self.weights = self._init_weights()

        # Input data
        self.input         = tf.placeholder(tf.float32, [None, self.n_input], name="input")
        # Boolean mask selecting the invalid (missing) values in the input data
        self.input_mask    = tf.placeholder(tf.bool, [None, self.n_input], name="input_mask") # True when input is invalid

        # Store the parameters in tensorflow constants.
        # dropout probability is converted to keep probability to simplify calculations below
        self.keep_prob     = tf.placeholder(tf.float32, shape=[], name="keep_probability")
        self.gaussian_prob = tf.placeholder(tf.float32, shape=[], name="gaussian_noise_probability")
        self.sap_prob      = tf.placeholder(tf.float32, shape=[], name="SAP_probability")
        self.alpha         = tf.constant(alpha_weight, dtype=tf.float32, shape=[], name="alpha")
        self.beta          = tf.constant(beta_weight, dtype=tf.float32, shape=[], name="beta")
        self.regl          = tf.constant(regl_weight, dtype=tf.float32, shape=[], name="regularization")
        self.rseed         = rseed

        input_shape = self.input.get_shape()

        # Masking noise mask is computed here since we need the binary tensor for
        # computing the loss function. The procedure is same as for gaussian noise.
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = self.keep_prob + tf.random_uniform(tf.shape(self.input), seed=self.get_rseed(), dtype=tf.float32)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)

        # Apply noise to the input.
        with tf.name_scope("noise"):
            self.noisy_input = self.add_gaussian_noise(self.input, self.gaussian_prob, gaussian_std)
            self.noisy_input = self.add_salt_and_pepper_noise(self.noisy_input, self.sap_prob)
            # Finish applying masking noise. Regularization is performed
            self.noisy_input = tf.div(self.noisy_input, self.keep_prob) * binary_tensor
            self.noisy_input.set_shape(input_shape)

        # Calculate the alpha mask (true for dropped out ratings) and beta mask
        # (true for non-dropped out ratings)
        with tf.name_scope("masks"):
            # True if a rating is valid and it has been dropped out
            self.alpha_mask = tf.logical_and(tf.equal(binary_tensor, 0),
                                             tf.logical_not(self.input_mask))
            self.alpha_mask.set_shape(input_shape)
            # True if a rating is valid and it has not been dropped out
            self.beta_mask = tf.logical_and(tf.equal(binary_tensor, 1),
                                            tf.logical_not(self.input_mask))
            self.beta_mask.set_shape(input_shape)
            
            tf.summary.scalar("alpha_mask_size",
                tf.reduce_sum(tf.cast(self.alpha_mask, tf.int32)),
                collections=[self.name])
            tf.summary.scalar("beta_mask_size",
                tf.reduce_sum(tf.cast(self.beta_mask, tf.int32)),
                collections=[self.name])
        
        # Calculate hidden and reconstruction layers of the NN
        with tf.name_scope("activations"):
            # Encoder
            self.hidden = tf.nn.tanh(tf.add(
                            tf.matmul(self.noisy_input, self.weights['W1']),
                            self.weights['b1']))
            # Decoder
            self.reconstruction = tf.nn.tanh(tf.add(
                            tf.matmul(self.hidden, self.weights['W2']), 
                            self.weights['b2']))
            # Summaries
            tf.summary.histogram('reconstruction', 
                tf.boolean_mask(self.reconstruction, 
                                tf.logical_not(self.input_mask)), collections=[self.name])
            tf.summary.histogram('hidden', self.hidden, collections=[self.name])
            tf.summary.histogram('input',
                tf.boolean_mask(self.input,
                                tf.logical_not(self.input_mask)), collections=[self.name])

        # Calculate the different components of the loss function
        with tf.name_scope("loss"):
            # Calculate loss
            self.alpha_cost = tf.reduce_sum(tf.squared_difference(
                                    tf.boolean_mask(self.reconstruction, self.alpha_mask), 
                                    tf.boolean_mask(self.input, self.alpha_mask))) * self.alpha
            self.beta_cost  = tf.reduce_sum(tf.squared_difference(
                                    tf.boolean_mask(self.reconstruction, self.beta_mask),
                                    tf.boolean_mask(self.input, self.beta_mask))) * self.beta
            # Apply regularization on the magnitude of the weight matrices
            self.regl_cost = self.regl * tf.add(tf.nn.l2_loss(self.weights['W1']),
                                                tf.nn.l2_loss(self.weights['W2']))
            self.cost = self.alpha_cost + self.beta_cost + self.regl_cost
 
            tf.summary.scalar('alpha_cost', self.alpha_cost, collections=[self.name])
            tf.summary.scalar('beta_cost', self.beta_cost, collections=[self.name])
            tf.summary.scalar('regl_cost', self.regl_cost, collections=[self.name])
            tf.summary.scalar('cost', self.cost, collections=[self.name])

        # Calculate the accuracy (RMSE) on validation data.
        # This is not executed at every iteration step, only when requested for progress updates
        with tf.name_scope("accuracy"):
            self.train_err = self._rmse(self.reconstruction, self.input, self.input_mask, "train_predictions")
            tf.summary.scalar("train_error", self.train_err, collections=[self.name])

            self.test_input = tf.placeholder(tf.float32, [None, self.n_input],
                    name="test_input")
            self.test_input_mask = tf.placeholder(tf.bool, [None, self.n_input], 
                    name="test_input_mask") # True when input is invalid
            self.test_err = self._rmse(self.reconstruction, self.test_input, self.test_input_mask, "test_predictions")
            tf.summary.scalar("test_error", self.test_err, collections=[self.name])

        with tf.name_scope("optimizer"):
            self.optimizer = optimizer.minimize(self.cost, global_step=batch)

        self.summary_op = tf.summary.merge_all(key=self.name)


    def add_gaussian_noise(self, data, noise_prob, std):
        """ Add random gaussian noise to a fraction of the data
        Algorithm:
        The choice of which elements to apply the noise is done by sampling
        a tensor from a uniform distribution in range [noise_prob, 1 + noise_prob)
        then with a floor operation this is converted to a binary tensor where
        there are on average the correct amount of ones (defined by noise_prob).
        Finally the a tensor containing gaussian noise is multiplied to the binary tensor
        to select only the correct portion of elements, and then added to the data tensor.
        """
        if noise_prob == 0.0:
            return data
        random_tensor = noise_prob
        random_tensor += tf.random_uniform(shape=tf.shape(data), seed=self.get_rseed(), dtype=data.dtype)
        binary_tensor = tf.floor(random_tensor)
        noise_tensor = tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=std, seed=self.get_rseed(), dtype=data.dtype) * \
                        binary_tensor
        return data + noise_tensor


    def add_salt_and_pepper_noise(self, X, noise_prob):
        """ Add random salt and pepper noise to a fraction of the data
        """
        if noise_prob == 0.0:
            return X
        mn = tf.reduce_min(X)
        mx = tf.reduce_max(X)

        half_noise_prob = 1 - noise_prob/2.0

        min_tensor = tf.floor(half_noise_prob + 
                            tf.random_uniform(shape=tf.shape(X), seed=self.get_rseed(), dtype=X.dtype))
        min_indices = tf.to_int32(
            tf.where(tf.equal(min_tensor, tf.constant(1, dtype=X.dtype))))
        num_true = tf.shape(min_indices)[0]

        X = tf.add(tf.multiply(X, min_tensor),
                   tf.scatter_nd(min_indices, tf.fill([num_true], mn), shape=tf.shape(X)))

        max_tensor = tf.floor(half_noise_prob + 
                            tf.random_uniform(shape=tf.shape(X), seed=self.get_rseed(), dtype=X.dtype))
        max_indices = tf.to_int32(
            tf.where(tf.equal(max_tensor, tf.constant(1, dtype=X.dtype))))
        num_true = tf.shape(max_indices)[0]

        X = tf.add(tf.multiply(X, max_tensor),
                   tf.scatter_nd(max_indices, tf.fill([num_true], mx), shape=tf.shape(X)))

        return X


    def _init_weights(self):
        """Initialize the input->hidden and hidden->reconstruction weight matrixes
        Also a bias vector is included in the weights.
        """
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
      """Add different summaries related to a variable"""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean, collections=[self.name])
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev, collections=[self.name])
        tf.summary.scalar('max/' + name, tf.reduce_max(var), collections=[self.name])
        tf.summary.scalar('min/' + name, tf.reduce_min(var), collections=[self.name])
        tf.summary.histogram(name, var, collections=[self.name])


    def fit(self, sess, input, input_mask):
        """Perform a single iteration of back-propagation. """
        fd = {
            self.input:         input,
            self.input_mask:    input_mask,
            self.keep_prob:     self.keep_prob_val,
            self.gaussian_prob: self.gaussian_prob_val,
            self.sap_prob:      self.sap_prob_val,
        }
        cost, opt = sess.run((self.cost, self.optimizer), feed_dict = fd)
        return cost


    def fit_summary(self, sess, input, input_mask, test, test_mask):
        """Obtain summaries and accuracy results. 
        For accuracy results we need to feed the validation data as well
        """
        fd = {
            self.input:           input,
            self.input_mask:      input_mask,
            self.test_input:      test,
            self.test_input_mask: test_mask,
            self.keep_prob:       self.keep_prob_val,
            self.gaussian_prob:   self.gaussian_prob_val,
            self.sap_prob:        self.sap_prob_val,
        }

        cost, trerr, tserr, summary = sess.run(
            (self.cost, self.train_err, self.test_err, self.summary_op), feed_dict = fd)
        return cost, trerr, tserr, summary


    def predictions(self, sess, input):
        """Make predictions without back-propagation. """

        fd = {
            self.input:         input,
            self.keep_prob:     1.0,
            self.gaussian_prob: 0.0,
            self.sap_prob:      0.0,
        }
        predictions = sess.run((self.reconstruction), feed_dict=fd)
        return predictions


    def hidden_activations(self, sess, input):
        """ Obtain the values of the hidden layer, without back-propagation"""
        fd = {
            self.input:         input,
            self.keep_prob:     self.keep_prob_val,
            self.gaussian_prob: self.gaussian_prob_val,
            self.sap_prob:      self.sap_prob_val,
        }
        hid = sess.run((self.hidden), feed_dict=fd)
        return hid


    def init_saver(self, more_vars, save_file):
        """ Add support for saving model weights.
        This should be called externally in order to specify additional variables
        which are to be saved.
        """
        var_list = list(self.weights.values()) + more_vars
        self.saver = tf.train.Saver(var_list=var_list)
        self.saver_filename = save_file


    def save_model(self, sess, global_step=None):
        return self.saver.save(sess, self.saver_filename, global_step=global_step)


    def restore_model(self, sess):
        return self.saver.restore(sess, self.saver_filename)


    def _rmse(self, preds, truth, mask, name):
        """Calculate the RMSE of `preds` with respect to `truth`.
        @param preds: predictions
        @param truth: true values for which predictions were made
        @param mask: boolean mask should be true when a value in the data is 
                invalid (not to be considered)
        @param name: how to name summaries related to the error
        """
        notmask = tf.logical_not(mask)
        valid_preds = tf.boolean_mask(preds, notmask)
        valid_preds = valid_preds*2 + 3
        tf.summary.histogram(name, valid_preds, collections=[self.name])

        valid_truth = tf.boolean_mask(truth, notmask)
        valid_truth = valid_truth*2 + 3
        tf.summary.histogram(name + "_truth", valid_truth, collections=[self.name])
        err = tf.sqrt(tf.reduce_mean(tf.squared_difference(valid_truth, valid_preds)))
        return err


    def get_rseed(self):
        """Retrieve random seed.
        To help obtain deterministic results we can use random seed. 
        However, always using the same seed will cause problems with noise being
        always applied to the same elements. Thus this function
        """
        if self.rseed is None:
            return None
        self.rseed += 1
        return self.rseed
