/**
 * Common type definitions used elsewhere in the project.
 *
 * author: gmeanti
 */
#ifndef __SGD_TYPES_H
#define __SGD_TYPES_H

#include <unordered_map>   // settings hashmap
#include <cstdint>         // int8_t
#include <utility>         // pair
#include <string>

#include <Eigen/Dense>

namespace reccommend {
    /* Main data type used to represent ratings (not input).
     * Defined as float because double is slower and does not give better results */
    typedef float dtype;
    /* Dynamically sized matrix of floating point */
    typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixD;
    /* Dynamically sized matrix of integers */
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixI;
    /* Pair of train-test input (integer) matrices */
    typedef std::pair<MatrixI, MatrixI> DataPair;
    /* Useful vectors */
    typedef Eigen::Matrix<dtype,  1, Eigen::Dynamic> RowVectorD;
    typedef Eigen::Matrix<dtype,  Eigen::Dynamic, 1> ColVectorD;
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, 1> ColVectorI;
    /* Settings are a hashmap from setting names to setting values */
    typedef std::unordered_map<std::string, dtype> Settings;
    typedef std::pair<std::string, dtype> SettingsEntry;
}

#endif
