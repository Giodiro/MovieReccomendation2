// uncomment to disable assert()
// #define NDEBUG
// #define EIGEN_NO_DEBUG
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
using reccommend::MatrixD;
using reccommend::Settings;
using reccommend::DataPair;
using reccommend::IOUtil;

const static int NUSERS = 10000;
const static int NITEMS = 1000;
const static std::string TRAIN_DATA_FILE = "../data_train.csv";
const static std::string MASK_DATA_FILE = "../sampleSubmission.csv";

/**
 * These settings were chosen via a long parameter search with the IntegratedSolver.
 */
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


void runAllClassif(const int num_threads, const ulong rseed,
                   const std::string submission_file,
                   const std::string mask_file,
                   const std::string train_file,
                   const int nusers, const int nitems,
                   const bool doPrediction);


void runAllClassif(const int num_threads, const ulong rseed,
                   const std::string submission_file,
                   const std::string mask_file,
                   const std::string train_file,
                   const int nusers, const int nitems,
                   const bool doPrediction)
{
    Settings settings = Settings(DEFAULT_SETTINGS);
    settings["num_threads"] = num_threads;

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


    std::cout << "******* Running SVD *******\n";

    test_score = reccommend::kfoldCV<reccommend::SVD>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver0 = reccommend::SVD(settings, data.first, mask);
        solver0.run();
        auto predictors = solver0.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SVD_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SVD_test.csv");
        trainVec.push_back(solver0.trainPredictor());
        testVec.push_back(solver0.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running SimpleSGDSolver *******\n";
    
    test_score = reccommend::kfoldCV<reccommend::SimpleSGDSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";
    if (doPrediction) {
        auto solver1 = reccommend::SimpleSGDSolver(settings, data.first, mask);
        solver1.run();
        auto predictors = solver1.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_simple_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_simple_test.csv");
        trainVec.push_back(solver1.trainPredictor());
        testVec.push_back(solver1.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running SGDppSolver *******\n";

    test_score = reccommend::kfoldCV<reccommend::SGDppSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver2 = reccommend::SGDppSolver(settings, data.first, mask);
        solver2.run();
        auto predictors = solver2.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_SGD++_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_SGD++_test.csv");
        trainVec.push_back(solver2.trainPredictor());
        testVec.push_back(solver2.testPredictor());
    }
    std::cout << "\n\n\n\n";

    std::cout << "******* Running IntegratedSolver *******\n";

    test_score = reccommend::kfoldCV<reccommend::IntegratedSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver3 = reccommend::IntegratedSolver(settings, data.first, mask);
        solver3.run();
        auto predictors = solver3.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Integrated_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Integrated_test.csv");
        trainVec.push_back(solver3.trainPredictor());
        testVec.push_back(solver3.testPredictor());
        std::cout << "\n\n\n\n";
    }
    std::cout << "******* Running NeighbourhoodSolver *******\n";
        
    test_score = reccommend::kfoldCV<reccommend::NeighbourhoodSolver>(k, settings, cv_data, 2);
    std::cout << "test score: " << test_score << "\n";

    if (doPrediction) {
        auto solver4 = reccommend::NeighbourhoodSolver(settings, data.first, mask);
        solver4.run();
        auto predictors = solver4.predictors();
        IOUtil::predictorToFile(data.first, predictors.first, submission_file + "_Neighbourhood_train.csv");
        IOUtil::predictorToFile(mask, predictors.second, submission_file + "_Neighbourhood_test.csv");
        trainVec.push_back(solver4.trainPredictor());
        testVec.push_back(solver4.testPredictor());
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
        runAllClassif(num_threads, rseed, "../saved_data/submissions/",
                      MASK_DATA_FILE, TRAIN_DATA_FILE,
                      NUSERS, NITEMS, true);
    }
    else if (benchmarkType == "movielens_1m") {
        runAllClassif(num_threads, rseed, "../saved_data/submissions/",
                      ".", "../saved_data/movielens/ml-1m/ratings_clean.csv",
                      6040, 3952, false);
    }
    else {
        std::cout << "Benchmark type " << benchmarkType << " is not implemented\n";
    }

    return 0;
}

