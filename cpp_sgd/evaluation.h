/**
 * Helper functions related to evaluation of solver algorithms.
 * Currently only 2 functions are implemented:
 *  - getScores is very simple and obtains the scores of training and validation predictors in a solver.
 *  - kfoldCV performs k-fold cross validation with a specified solver class.
 */
#ifndef __EVALUATION_H
#define __EVALUATION_H
#include <utility>      // pair
#include <vector>

#include "sgd.h"        // SGDSolver
#include "sgd_types.h"

namespace reccommend {

    /*
     * Simple wrapper to obtain train and test data predictors of the solvers,
     * and calculate their (RMSE) score. First element in the pair is training score,
     * second element is the test score.
     */
    std::pair<double, double> getScores(SGDSolver &solver);

    /*
     * Perform k-fold cross validation with a specific solver.
     * Returns the mean score obtained on the k validation datasets.
     * Increasing verbosity allows to print other statistics 
     * (such as the standard deviation of the scores for example).
     */
    template <class Solver>
    double kfoldCV(const int k, Settings &config, 
                   const std::vector< DataPair > &cvData,
                   const int verbose);

}
#endif