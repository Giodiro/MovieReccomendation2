

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

