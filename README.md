## Folder organization

 - *sgd_cpp*: contains the C++ implementation of stochastic gradient descent and many algorithms that use it for collaborative filtering. 2 main programs can be compiled: the first (`SGD`) uses all available algorithms and performs predictions writing them to file; the second (`MPI_SGD`) is used for performing random parameter search.
 - *autoencoder*: implementation of a denoising autoencoder for collaborative filtering. Main file in here is `tf_sgd.py`.
 - *unused_python*: contains old implementations, useful for reference but not used in practice anymore.
 - *analyze_data.R*: script for analyzing results of parameter search.
 - *XGBoost.ipynb*: IPython notebook used for running xgboost on many predictors obtained from the autoencoder and the SGD. The results of this are used for final predictions.

## Required libraries for training the autoencoder (`tf_sgd.py`)
 
 The autoencoder uses the tensorflow library, along with numpy and possibly other python 3rd party libraries.

## Required libraries for running `SGD` and `MPI_SGD`
 
 - Eigen
 - OpenMP
 - MPI (only for running the parameter search)
 - xgboost (which depends on a number of other things)

### Installing Eigen

 1. Download the header files from http://eigen.tuxfamily.org/index.php?title=Main_Page#Download
 2. Extract the files and copy the "Eigen" folder to a folder named "lib" at the top level of the repository.

For more detailed information see https://eigen.tuxfamily.org/dox/GettingStarted.html 

### Installing XGBoost

 XGBoost is much easier to use from python, therefore it is not necessary to install and compile it externally. Just use `pip install xgboost`.

### Installing OpenMP

 OpenMP does not need installing.

### Installing OpenMPI

 On Ubuntu it is enough to run
 `sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev`

