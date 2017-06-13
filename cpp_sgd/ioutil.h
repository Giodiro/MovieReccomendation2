#ifndef __IOUtil_H
#define __IOUtil_H

#include <string>
#include <Eigen/Dense>
#include <utility>      // pair
#include <cstdint>      // int8_t

#include <ctime>
#include <chrono>       // time measurements
#include <sstream>      // ostringstream
#include <iomanip>      // put_time

#include "sgd_types.h"

namespace reccommend {
    /*
     * Static class for reading data from file, and writing to file.
     */
    class IOUtil
    {
        private:
            IOUtil();
            struct MatrixEntry {
              int row, col, val;
            };
            static MatrixEntry readLine(std::string &line);
        public:
            static DataPair readData(std::string fileName, double testPercentage, int nusers, 
                             int nitems, unsigned long seed);
            static MatrixI readMask(const std::string fileName, int nusers, int nitems);

            /*
             * Read data from specified file,
             * subdivide the data into k training and validation sets for k-fold cross validation
             * Parameters include the number of folds (usually 3 or 5) and the percentage of data to
             * include in the validation sets (usefull for e.g. 3-fold CV when taking 33% of the data 
             * for validation is too much).
             */
            static std::vector < std::pair < MatrixI, MatrixI > > readDataCV (
                    std::string filename,
                    int k_cv,
                    int nusers,
                    int nitems,
                    float test_percentage,
                    unsigned long seed);

            /*
             * Write a matrix of predictions to file. 
             * The first parameter "entries" contains non-zero entries 
             * in the positions where a prediction must be made.
             * The "predictions" parameter contains the actual predictions.
             */
            static void predictorToFile(const MatrixI &entries,
                                        const MatrixD &predictions,
                                        const std::string submission_file);
    };

    /*
     * Following functions are utilities for displaying elapsed time or current time.
     */

    double inline elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start) 
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    std::string inline now()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "[%H:%M:%S] ");
        return oss.str();
    }
}

#endif
