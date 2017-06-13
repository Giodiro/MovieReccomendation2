#ifndef __SGD_UTIL_H
#define __SGD_UTIL_H

#include "sgd_types.h"
#include <utility>      // pair
#include <vector>       // vector


reccommend::dtype _calcPearson(const int i1, const int i2, 
                               const int lambda2, const reccommend::MatrixI &data);


namespace reccommend {

    MatrixD calcPearsonMatrix (const MatrixI &data, 
                               const int shrinkage, 
                               const int num_threads);

    dtype   calcGlobalMean    (const MatrixI &data, 
                               const std::vector<std::pair<int, int> > &dataIndices);

    /**
       Calculate the per-user and per-item bias matrix. 
       The matrix is used to assign a default value to the missing ratings 
       in the training matrix.
    */
    MatrixD calcBiasMatrix    (const MatrixI &data, 
                               const dtype globalMean, 
                               const int K1, 
                               const int K2);
}



#endif