#include <string>
#include <iostream>
#include <fstream>      // std::ofstream
#include <memory>
#include <chrono>       // time measurements
#include <random>
#include <iterator>
#include <stdexcept>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

#include <Eigen/Dense>

// Include the xgboost API
#include <xgboost/c_api.h>

#include "sgd_types.h"
#include "ioutil.h"
#include "evaluation.h"
#include "sgd.h"

using reccommend::dtype;
using reccommend::MatrixI;
using reccommend::MatrixD;
using reccommend::Settings;
using reccommend::DataPair;
using reccommend::IOUtil;

const static int NUSERS = 10000;
const static int NITEMS = 1000;
const static std::string TRAIN_DATA_FILE = "../data_train.csv";
const static std::string MASK_DATA_FILE = "../sampleSubmission.csv";

const static Settings DEFAULT_SETTINGS = {
    {"nusers", NUSERS},
    {"nitems", NITEMS},
    {"lrate1", 0.006},
    {"lrate2", 0.00546573},
    {"lrate3", 0.00525803},
    {"regl4", 0.0360761},
    {"regl6", 0.0412977},
    {"regl7", 0.0698449},
    {"regl8", 0.0542553},
    {"lrate_reduction", 0.96},
    {"num_factors", 21},
    {"max_iter", 51},
    {"correlation_shrinkage", 100},
    {"K1", 1},
    {"K2", 10},
    {"max_neigh", 264},
};


// XGBoost functions
void BuildXGBoostMatrix (std::vector< std::vector<dtype> > &predictors, DMatrixHandle *out);
void RunXGBoost (std::vector< std::vector<dtype> > &trp, 
    std::vector< std::vector<dtype> > &tsp, 
    const MatrixD &train, 
    const MatrixD &test,
    const std::string submission_file,
    const bool save_data,
    const bool load_data);
void xgboostClassifDriver(int num_threads, ulong rseed, std::string submission_file);



void BuildXGBoostMatrix (std::vector< std::vector<dtype> > &predictors, DMatrixHandle *out) 
{
    size_t ncol = predictors.size();
    size_t nrow = predictors[0].size();
    std::cout << "nrow: " << nrow << " ncol: " << ncol << "\n";
    float *data = static_cast<float *>(malloc(nrow * ncol * sizeof(float)));
    uint ti = 0;
    for (uint i = 0; i < nrow; ++i) {
        for (uint j = 0; j < ncol; ++j) {
            data[ti] = predictors[j][i];
            ti++;
        }
    }
    XGDMatrixCreateFromMat(data, nrow, ncol, -1, out);
    free(data);
}


void RunXGBoost (std::vector< std::vector<dtype> > &trp, 
                 std::vector< std::vector<dtype> > &tsp, 
                 const MatrixD &train, 
                 const MatrixD &test,
                 const std::string submission_file,
                 const bool save_data,
                 const bool load_data)
{
    int res;
    DMatrixHandle h_train[1];
    DMatrixHandle h_test;
    const char* evnames[1]; // Names of the data matrix
    evnames[0] = "training";

    if (load_data) {
        res = XGDMatrixCreateFromFile("train_matrix.dat", 0, &h_train[0]);
        if (res != 0) {
            std::cout << "Failed to load training matrix. Exiting\n";
            return;
        }
        res = XGDMatrixCreateFromFile("test_matrix.dat", 0, &h_test);
        if (res != 0) {
            std::cout << "Failed to load test matrix. Exiting\n";
            return;
        }
    } else {
        BuildXGBoostMatrix(trp, &h_train[0]);
        BuildXGBoostMatrix(tsp, &h_test);

        if (save_data) {
            res = XGDMatrixSaveBinary(h_train[0], "train_matrix.dat", 0);
            if (res != 0) {
                std::cout << "Failed to save training matrix\n";
            }
            res = XGDMatrixSaveBinary(h_test, "test_matrix.dat", 0);
            if (res != 0) {
                std::cout << "Failed to save test matrix\n";
            }
        }
    }
    
    // Labels:
    double global_bias = 0;
    auto train_labels = std::vector<float>(trp[0].size());
    int ti = 0;
    for (int i = 0; i < train.rows(); ++i) {
        for (int j = 0; j < train.cols(); ++j) {
            if (train(i, j) > 0) {
                train_labels[ti] = train(i, j);
                global_bias += train(i, j);
                ti++;
            }
        }
    }
    global_bias /= ti;

    XGDMatrixSetFloatInfo(h_train[0], "label", train_labels.data(), trp[0].size());

    BoosterHandle h_booster;
    XGBoosterCreate(h_train, 1, &h_booster);
    XGBoosterSetParam(h_booster, "booster", "gbtree");
    XGBoosterSetParam(h_booster, "objective", "reg:linear");
    XGBoosterSetParam(h_booster, "eval_metric", "rmse");
    XGBoosterSetParam(h_booster, "seed", "123921");
    XGBoosterSetParam(h_booster, "eta", "0.006");
    XGBoosterSetParam(h_booster, "max_depth", "3");
    XGBoosterSetParam(h_booster, "alpha", "10");
    XGBoosterSetParam(h_booster, "lambda", "10");
    XGBoosterSetParam(h_booster, "subsample", "0.5");
    XGBoosterSetParam(h_booster, "base_score", std::to_string(global_bias).c_str());

    // TODO: Early stopping?
    int num_iter = 200;
    int report_every = 10;
    for (int iter = 1; iter <= num_iter; iter++) {
        if (iter % report_every == 0) {
            std::cout << "boosting round " << iter << "\n";
            const char *eval_summary; // Memory leak
            res = XGBoosterEvalOneIter(h_booster, iter, h_train, evnames, 1, &eval_summary);
            if (res == 0) {
                std::cout << eval_summary << "\n";
            }
        }
        XGBoosterUpdateOneIter(h_booster, iter, h_train[0]);
    }

    // Predict
    bst_ulong out_len;
    const float *f; // This is never freed! (Will leak but who cares?)
    XGBoosterPredict(h_booster, h_test, 0, 0, &out_len, &f);

    // Write the 1D array to submission file. 
    // Must translate back from 1D array to the 2D format
    std::ofstream ofs;
    ofs.open (submission_file, std::ofstream::out | std::ofstream::trunc);
    ofs << "Id,Prediction\n";
    uint fit = 0;
    for (int u = 0; u < test.rows(); ++u) {
        for (int i = 0; i < test.cols(); ++i) {
            if (test(u, i) > 0) {
                assert (fit < out_len);
                ofs << "r" << (u + 1) << "_c" << (i + 1) << "," << f[fit] << "\n";
                fit++;
            }
        }
    }
    ofs.close();

    // free xgboost internal structures
    XGDMatrixFree(h_train[0]);
    XGDMatrixFree(h_test);
    XGBoosterFree(h_booster);
}


void xgboostClassifDriver(int num_threads, ulong rseed, std::string submission_file) 
{
    Settings settings = Settings(DEFAULT_SETTINGS);
    settings["num_threads"] = num_threads;

    int k = 3;
    MatrixI mask;
    DataPair data;
    std::vector<DataPair> cv_data;
    try {
        mask = IOUtil::readMask(MASK_DATA_FILE, NUSERS, NITEMS);
        data = IOUtil::readData(TRAIN_DATA_FILE, 0.0, NUSERS, NITEMS, rseed);
        cv_data = IOUtil::readDataCV(TRAIN_DATA_FILE, k, NUSERS, NITEMS, 0.2, rseed);
    } 
    catch (const std::runtime_error& e) {
        cerr << e.what() << "\n";
        cerr << "Aborting\n";
        return;
    }
    double test_score;

    auto trainVec = std::vector< std::vector<dtype> >();
    auto testVec = std::vector< std::vector<dtype> >();

    std::cout << "******* Running SVD *******\n";

    test_score = reccommend::kfoldCV<reccommend::SVD>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    auto solver0 = reccommend::SVD(settings, data.first, mask);
    solver0.run();
    auto predictors = solver0.predictors();
    IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SVD_train.csv");
    IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SVD_test.csv");
    trainVec.push_back(solver0.trainPredictor());
    testVec.push_back(solver0.testPredictor());
    std::cout << "\n\n\n\n";


    std::cout << "******* Running SimpleSGDSolver *******\n";
    
    test_score = reccommend::kfoldCV<reccommend::SimpleSGDSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    auto solver1 = reccommend::SimpleSGDSolver(settings, data.first, mask);
    solver1.run();
    predictors = solver1.predictors();
    IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_simple_train.csv");
    IOUtil::predictorToFile(mask, predictors.second, submission_file + "_simple_test.csv");
    trainVec.push_back(solver1.trainPredictor());
    testVec.push_back(solver1.testPredictor());
    std::cout << "\n\n\n\n";

    std::cout << "******* Running SGDppSolver *******\n";

    test_score = reccommend::kfoldCV<reccommend::SGDppSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";


    auto solver2 = reccommend::SGDppSolver(settings, data.first, mask);
    solver2.run();
    predictors = solver2.predictors();
    IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SGD++_train.csv");
    IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SGD++_test.csv");
    trainVec.push_back(solver2.trainPredictor());
    testVec.push_back(solver2.testPredictor());
    std::cout << "\n\n\n\n";

    std::cout << "******* Running IntegratedSolver *******\n";

    test_score = reccommend::kfoldCV<reccommend::IntegratedSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    auto solver3 = reccommend::IntegratedSolver(settings, data.first, mask);
    solver3.run();
    predictors = solver3.predictors();
    IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Integrated_train.csv");
    IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Integrated_test.csv");
    trainVec.push_back(solver3.trainPredictor());
    testVec.push_back(solver3.testPredictor());
    std::cout << "\n\n\n\n";

    std::cout << "******* Running NeighbourhoodSolver *******\n";
        
    test_score = reccommend::kfoldCV<reccommend::NeighbourhoodSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    auto solver4 = reccommend::NeighbourhoodSolver(settings, data.first, mask);
    solver4.run();
    predictors = solver4.predictors();
    IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Neighbourhood_train.csv");
    IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Neighbourhood_test.csv");
    trainVec.push_back(solver4.trainPredictor());
    testVec.push_back(solver4.testPredictor());
    std::cout << "\n\n\n\n";

    //RunXGBoost(trainVec, testVec, data.first.cast<dtype>(),
    //           mask.cast<dtype>(), submission_file, true, false);
}


int main () {
    // Force flushing of output
    std::cout.setf( std::ios_base::unitbuf );

    unsigned long rseed = 123589;

    int num_threads = 4;
    if (const char* env_p = std::getenv("OMP_NUM_THREADS")) {
        try {
            num_threads = std::stoi(env_p);
        } catch (const std::invalid_argument &ia) { }
    }

    xgboostClassifDriver(num_threads, rseed, "submit_xgboost.csv");

    return 0;
}

