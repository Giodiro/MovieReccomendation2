## Rough code guide:

- _MaskedDenoisingAutoencoder.py_: contains the class which implements the autoencoder in tensorflow
- _StackedAutoencoder.py_: implements a stacked autoencoder in tensorflow. This model was not used for final predictions due to poor performance and the code may not work anymore.
- _AutoencoderRunner.py_: point of entry for training the autoencoder. Run python `AutoencoderRunner`. To make predictions it is necessary to modify the parameters within the file (command line arguments are not implemented)
- _BayesianOptimization.py_: hyperparameter search for the autoencoder
- _util.py_: utility functions