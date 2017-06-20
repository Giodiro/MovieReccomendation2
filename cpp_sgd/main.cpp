/**  main.cpp
 * The main program implemented here uses all solvers implemented in this project
 * in order to obtain files ready for submission.
 * For each solver we perform 3-fold CV to have a rough idea of how it will perform
 * and then use all of the training data to obtain a "predictor" file.
 *
 * This file has an optional command line argument to specify different datasets. Currently the two options are:
 * - "submission": default option which runs the algorithms on the in-class supplied data, and outputs predictor files
 * - "movielens": runs on the movielens 1m dataset (available https://grouplens.org/datasets/movielens/), 
 *                cleaned to resemble the in-class supplied files.
 */

// uncomment to disable assert()
#define NDEBUG
#define EIGEN_NO_DEBUG
#include <string>
#include <iostream>
#include <stdexcept>

#include <Eigen/Dense>

#include "sgd_types.h"
#include "ioutil.h"
#include "evaluation.h"
#include "sgd.h"

using reccommend::dtype;
using reccommend::MatrixI;
using reccommend::Settings;
using reccommend::DataPair;
using reccommend::IOUtil;

const static int NUSERS = 10000;
const static int NITEMS = 1000;
const static std::string TRAIN_DATA_FILE = "../data_train.csv";
const static std::string MASK_DATA_FILE  = "../sampleSubmission.csv";
const static std::string MOVIELENS_DIR   = "../saved_data/movielens/ml-1m/";
const static std::string SUBMISSION_DIR  = "../saved_data/submissions/";

/**
 * These settings were chosen via a long parameter search with the IntegratedSolver.
 * 3-fold CV score obtained was 0.976816
 */
const static Settings DEFAULT_SETTINGS = {
    {"nusers", NUSERS},
    {"nitems", NITEMS},
    {"lrate1", 0.0253247},
    {"lrate2", 0.024758},
    {"lrate3", 0.0118577},
    {"regl4", 0.196401},
    {"regl6", 0.135719},
    {"regl7", 0.0653609},
    {"regl8", 0.162456},
    {"lrate_reduction", 0.922435},
    {"num_factors", 44},
    {"correlation_shrinkage", 100}, // Not included in search
    {"K1", 1}, // Not included in search
    {"K2", 18}, // Not included in search (set equal to SVD_SETTINGS)
    {"max_neigh", 127},
    {"max_iter", 72},
};

/**
 * Settings chosen after SVD random parameter search.
 * 3-fold CV score obtained was: 0.988783
 */
const static Settings SVD_SETTINGS = {
    {"nusers", NUSERS},
    {"nitems", NITEMS},
    {"num_factors", 19},
    {"K1", 1},
    {"K2", 18},
    {"max_iter", 1}, // This setting is not used (not an iterative process), but needed here.
};

/**
 * Settings chosen with random parameter search.
 * 3-fold CV score obtained was: 0.980616
 */
const static Settings SIMPLE_SETTINGS = {
    {"nusers", NUSERS},
    {"nitems", NITEMS},
    {"lrate1", 0.00472091},
    {"lrate2", 0.0212446},
    {"regl6", 0.0463903},
    {"regl7", 0.0914689},
    {"lrate_reduction", 0.965086},
    {"num_factors", 96},
    {"max_iter", 78},
};

/**
 * Settings chosen with random parameter search.
 * 3-fold CV score obtained was: 0.979596
 */
const static Settings SGDpp_SETTINGS = {
    {"nusers", NUSERS},
    {"nitems", NITEMS},
    {"lrate1", 0.0068136},
    {"lrate2", 0.017407},
    {"regl6", 0.147678},
    {"regl7", 0.0583994},
    {"lrate_reduction", 0.925077},
    {"num_factors", 100},
    {"max_iter", 80},
};

/**
 * Returns a string explaining the program's command line arguments
 */
std::string usageString();

/**
 * Main function to run all the collaborative filtering algorithms.
 * TODO: This could use some refactoring to reduce code duplication
 */
void runAllClassif(const int num_threads, const ulong rseed,
                   const std::string submission_file,
                   const std::string mask_file,
                   const std::string train_file,
                   const int nusers, const int nitems,
                   const bool doPrediction);


/**** IMPLEMENTATION ****/


std::string usageString() {
    return "Usage: SGD [submission|movielens]";
}


void runAllClassif(const int num_threads, const ulong rseed,
                   const std::string submission_file,
                   const std::string mask_file,
                   const std::string train_file,
                   const int nusers, const int nitems,
                   const bool doPrediction)
{
    Settings settings = Settings(DEFAULT_SETTINGS);
    settings["num_threads"] = num_threads;

    Settings svdSettings = Settings(SVD_SETTINGS);
    svdSettings["num_threads"] = num_threads;

    Settings simpleSettings = Settings(SIMPLE_SETTINGS);
    simpleSettings["num_threads"] = num_threads;

    Settings sgdppSettings = Settings(SGDpp_SETTINGS);
    sgdppSettings["num_threads"] = num_threads;

    int k = 3;
    MatrixI mask;
    DataPair data;
    std::vector<DataPair> cv_data;
    try {
        mask = IOUtil::readMask(mask_file, nusers, nitems);
        data = IOUtil::readData(train_file, 0.0, nusers, nitems, rseed);
        cv_data = IOUtil::readDataCV(train_file, k, nusers, nitems, 0.2, rseed);
    } 
    catch (const std::runtime_error& e) {
        cerr << e.what() << "\n";
        cerr << "Aborting\n";
        return;
    }
    double test_score;

    auto trainVec = std::vector< std::vector<dtype> >();
    auto testVec = std::vector< std::vector<dtype> >();

    //std::vector< std::unique_ptr <reccommend::SGDSolver> > solverList;

    //solverList.emplace_back(std::unique_ptr<reccommend::SGDSolver>(new reccommend::SVD(settings, data.first, mask)));

    std::cout << "******* Running Baseline Predictor *******\n";

    // Use same settings as SVD settings (K1 and K2 are the only things needed here)
    test_score = reccommend::kfoldCV<reccommend::BiasPredictor>(k, svdSettings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::BiasPredictor(svdSettings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Bias_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Bias_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }
    std::cout << "\n\n\n\n";


    std::cout << "******* Running SVD *******\n";

    test_score = reccommend::kfoldCV<reccommend::SVD>(k, svdSettings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::SVD(svdSettings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SVD_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SVD_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running SimpleSGDSolver *******\n";
    
    test_score = reccommend::kfoldCV<reccommend::SimpleSGDSolver>(k, simpleSettings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";
    if (doPrediction) {
        auto solver = reccommend::SimpleSGDSolver(simpleSettings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_simple_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_simple_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running SGDppSolver *******\n";

    test_score = reccommend::kfoldCV<reccommend::SGDppSolver>(k, sgdppSettings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::SGDppSolver(sgdppSettings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SGD++_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SGD++_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running IntegratedSolver (pearson) *******\n";

    test_score = reccommend::kfoldCV<reccommend::IntegratedPearsonSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::IntegratedPearsonSolver(settings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Integrated_pearson_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Integrated_pearson_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
        std::cout << "\n\n\n\n";
    }

    std::cout << "******* Running IntegratedSolver (spearman) *******\n";

    test_score = reccommend::kfoldCV<reccommend::IntegratedSpearmanSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::IntegratedSpearmanSolver(settings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Integrated_spearman_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Integrated_spearman_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
        std::cout << "\n\n\n\n";
    }


    std::cout << "******* Running NeighbourhoodSolver (pearson) *******\n";

    test_score = reccommend::kfoldCV<reccommend::NeighbourhoodPearsonSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::NeighbourhoodPearsonSolver(settings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Neighbourhood_pearson_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Neighbourhood_pearson_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }

    std::cout << "******* Running NeighbourhoodSolver (spearman) *******\n";
        
    test_score = reccommend::kfoldCV<reccommend::NeighbourhoodSpearmanSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver = reccommend::NeighbourhoodSpearmanSolver(settings, data.first, mask);
        solver.run();
        auto predictors = solver.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Neighbourhood_spearman_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Neighbourhood_spearman_test.csv");
        trainVec.push_back(solver.trainPredictor());
        testVec.push_back(solver.testPredictor());
    }

    std::cout << "\n\nFINISHED\n";
}


int main(int argc, char** argv) {
    // Force flushing of output
    std::cout.setf( std::ios_base::unitbuf );

    unsigned long rseed = 123589;

    int num_threads = 4;
    if (const char* env_p = std::getenv("OMP_NUM_THREADS")) {
        try {
            num_threads = std::stoi(env_p);
        } catch (const std::invalid_argument &ia) { }
    }

    std::string benchmarkType;
    if (argc > 1) {
        benchmarkType = argv[1];
    } else {
        benchmarkType = "submission";
    }

    if (benchmarkType == "submission") {
        runAllClassif(num_threads, rseed, SUBMISSION_DIR,
                      MASK_DATA_FILE, TRAIN_DATA_FILE,
                      NUSERS, NITEMS, true);
    }
    else if (benchmarkType == "movielens_1m") {
        runAllClassif(num_threads, rseed, SUBMISSION_DIR,
                      ".", MOVIELENS_DIR + "ratings_clean.csv",
                      6040, 3952, false);
    }
    else {
        std::cout << "Benchmark type " << benchmarkType << " is not implemented\n";
        std::cout << usageString() << "\n";
    }

    return 0;
}

