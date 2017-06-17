/**
 * Implementation of ioutil.h
 *
 * author: gmeanti
 */
#include <iostream>
#include <random>     // std::mt19937
#include <fstream>    // std::ifstream, std::ofstream
#include <utility>    // std::pair
#include <stdexcept>  // std::runtime_error

#include "ioutil.h"

using std::string;
using std::pair;
using std::cout;
using reccommend::MatrixI;
using reccommend::MatrixD;
using reccommend::DataPair;

reccommend::IOUtil::MatrixEntry reccommend::IOUtil::readLine(string &line) {
    // Sample line (from data_train.csv):
    // r44_c1,4
    IOUtil::MatrixEntry result = {};

    size_t row_col_separator = line.find("_");
    result.row = stoi(line.substr(1, row_col_separator)) - 1;
    size_t comma = line.find(",", row_col_separator);
    result.col = stoi(line.substr(row_col_separator+2, comma)) - 1;
    result.val = stoi(line.substr(comma+1, line.length()));
    
    return result;
}

DataPair reccommend::IOUtil::readData(string fileName, double testPercentage, 
                                    int nusers, int nitems, unsigned long seed) 
{
    MatrixI train = MatrixI::Zero(nusers, nitems);
    MatrixI test = MatrixI::Zero(nusers, nitems);

    // Uniform [0, 1] generator. Only used if testPercentage > 0
    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> uni(0, 1);

    int ntrain = 0;
    int ntest = 0;

    std::ifstream fh (fileName, std::ifstream::in);
    string line;
    if (fh.is_open())
    {
        // Skip first line ("Id,Prediction")
        std::getline(fh, line);
        while ( std::getline (fh,line) )
        {
            MatrixEntry me = IOUtil::readLine(line);
            if (testPercentage > 0 && uni(gen) < testPercentage) {
                test(me.row, me.col) = me.val;
                ntest++;
            }
            else {
                train(me.row, me.col) = me.val;
                ntrain++;
            }
        }
        fh.close();
    }
    else {
        //cout << "Failed to read from file " << fileName << "\n";
        throw std::runtime_error("Failed to read from file " + fileName);
        //return pair<MatrixI, MatrixI>(train, test);
    }
    cout << "Read data from " << fileName << "\n";
    cout << "Training examples: " << ntrain << "; test examples: " << ntest << "\n";
    return pair<MatrixI, MatrixI>(train, test);
}


std::vector < DataPair > reccommend::IOUtil::readDataCV (
            string filename,
            int k_cv,
            int nusers,
            int nitems,
            float test_percentage, // This must be less than 1
            unsigned long seed)
{
    // Initialize data containers
    std::vector<DataPair> dataVec (k_cv);

    // Read the data into a single matrix
    auto initData = readData(filename, 0.0, nusers, nitems, seed).first;

    // Store the linearized coordinates so its easier to do random shuffle
    std::vector<int> index;
    for (int i = 0; i < nusers; i++) {
        for (int j = 0; j < nitems; j++) {
            if (initData(i, j) > 0) {
                index.push_back(i * nitems + j);
            }
        }
    }

    std::mt19937 gen(seed);
    std::shuffle(index.begin(), index.end(), gen);

    for (int i = 0; i < k_cv; i++) {
        dataVec[i] = pair<MatrixI, MatrixI>(initData, MatrixI::Zero(nusers, nitems));
        for (size_t tsi = i * (index.size() * test_percentage); tsi < (i+1) * (index.size() * test_percentage); tsi++) {
            int lini = index[tsi];
            // Swap from train matrix to test matrix
            std::swap(dataVec[i].first(lini / nitems, lini % nitems),
                      dataVec[i].second(lini / nitems, lini % nitems));
        }
    }
    std::cout << "CV uses " << index.size() * (1 - test_percentage) << " train and " << index.size() * test_percentage << " test examples\n";

    return dataVec;
}


MatrixI reccommend::IOUtil::readMask(string fileName, int nusers, int nitems) {
    MatrixI mask = MatrixI::Zero(nusers, nitems);

    std::ifstream fh (fileName, std::ifstream::in);
    string line;
    if (fh.is_open())
    {
        // Skip first line ("Id,Prediction")
        std::getline(fh, line);
        while ( std::getline (fh,line) )
        {
            MatrixEntry me = IOUtil::readLine(line);
            mask(me.row, me.col) = 1;
        }
        fh.close();
    }
    else {
        throw std::runtime_error("Failed to read from file " + fileName);
    }

    return mask;
}


void reccommend::IOUtil::predictorToFile(const MatrixI &entries,
                                         const MatrixD &predictions,
                                         const std::string submission_file)
{
    std::ofstream ofs;
    ofs.open (submission_file, std::ofstream::out | std::ofstream::trunc);
    int written = 0;
    ofs << "Id,Prediction\n";
    for (int u = 0; u < entries.rows(); ++u) {
        for (int i = 0; i < entries.cols(); ++i) {
            if (entries(u, i) > 0) {
                written++;
                ofs << "r" << (u + 1) << "_c" << (i + 1) << "," << predictions(u, i) << "\n";
            }
        }
    }
    std::cout << "Written " << written << " data points to " << submission_file << "\n";
    ofs.close();
}