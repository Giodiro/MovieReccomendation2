/**
 * A few utility functions used in the collaborative filtering solvers.
 * In particular most of the code is devoted to calculating correlation matrices (pearson and spearman).
 * There is also a simple function to compute the mean of a data matrix, and the per-user, per-item bias.
 *
 * author: gmeanti
 */
#ifndef __SGD_UTIL_H
#define __SGD_UTIL_H

#include <utility>      // pair
#include <vector>       // vector

#include "sgd_types.h"

/**
 * Calculate the Pearson correlation coefficient between columns i1, i2 in matrix data.
 * the lambda2 parameter is a shrinkage parameter for the correlation coefficient.
 */
reccommend::dtype _calcPearson(const int i1, const int i2, 
                               const int lambda2, const reccommend::MatrixI &data);
/**
 * Outputs the ranking of the input vector, using averaging to break ties
 * (see wikipedia article on ranking)
 */
reccommend::ColVectorD _rank(const reccommend::ColVectorI &v);

/**
 * Calculate the Spearman correlation coefficient between columns i1, i2 in the data matrix.
 * This coefficient is based on ranked data, so the function will use the _rank function above.
 */
reccommend::dtype _calcSpearman(const int i1, const int i2, const reccommend::MatrixI &data);


namespace reccommend {
    /**
     * Compute the Pearson correlation matrix between the items (columns) in the input matrix.
     * Parallel computations are possible.
     */
    MatrixD calcPearsonMatrix (const MatrixI &data,
                               const int shrinkage,
                               const int num_threads);

    /**
     * Compute the Spearman correlation matrix between the items (columns) in the input matrix.
     * Parallel computations are possible.
     */
    MatrixD calcSpearmanMatrix (const MatrixI &data,
                                const int num_threads);

    /**
     * Calculate the mean of the valid (non-zero) entries in the input matrix.
     */
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