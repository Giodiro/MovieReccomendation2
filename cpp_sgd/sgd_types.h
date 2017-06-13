#ifndef __SGD_TYPES_H
#define __SGD_TYPES_H

#include <unordered_map>   // settings hashmap
#include <cstdint>         // int8_t
#include <utility>         // pair
#include <string>

#include <Eigen/Dense>

namespace reccommend {

    typedef float dtype;
    typedef Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixD;


    typedef std::unordered_map<std::string, dtype> Settings;
    typedef std::pair<std::string, dtype> SettingsEntry;
    
    typedef Eigen::Matrix<dtype, 1, Eigen::Dynamic> RowVectorD;
    // Change name from VectorD to ColVectorD
    typedef Eigen::Matrix<dtype, Eigen::Dynamic, 1> ColVectorD;

    /* USELESS
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixTr;
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixTs;
    */


    // From ioutil.h
    typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixI;


    // from main.cpp
    /* Unneeded 
    typedef std::unordered_map<std::string, float> dconfig;
    */

    typedef std::pair<MatrixI, MatrixI> DataPair;

}

#endif
