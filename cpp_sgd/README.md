## Rough code guide:

- _main.cpp_: main point of entry for making predictions with all implemented models. Compile with `make sgd`, and run `SGD`.
- _mpi_main.cpp_: main point of entry for running a parameter search. Command line parameters define which model will be used for the search. Compile with `make mpi` and run with `MPI_SGD`.
- _sgd.cpp_: Implementation of all the models
- _sgd_types.h_: typedefs used in the rest of the code
- _variant.cpp_: implementation (taken from the internet, not my code!) of a variant type used to define bounds for parameters in the parameter search.
- _evaluation.cpp_, _ioutil.cpp_, _sgd\_util.cpp_: implement utility functions used elsewhere