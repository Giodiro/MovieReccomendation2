/**
 * Implementation file for evaluation.h
 *
 * author: gmeanti
 */
#include <iostream>
#include <chrono>

#include "evaluation.h"
#include "ioutil.h"        // for the timing functions

using reccommend::DataPair;
using reccommend::Settings;
using reccommend::SGDSolver;

template <class Solver>
double reccommend::kfoldCV(const int k, 
                           Settings &config, 
                           const std::vector< DataPair > &cvData,
                           const int verbose)
{
    assert((cvData.size() >= static_cast<size_t>(k)));

    double mtrScore = 0, mtsScore = 0; // mean of scores
    double strScore = 0, stsScore = 0; // std of scores

    std::vector<double> trainScores(k);
    std::vector<double> testScores(k);

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < k; i++) {
        if (verbose > 0) {
            std::cout << reccommend::now() << "Started " << k << "-fold iteration " << i << "\n";
        }
        Solver solver(config, cvData[i].first, cvData[i].second);
        solver.run();

        std::pair<double, double> scores = reccommend::getScores(solver);
        trainScores[i] = scores.first;
        testScores[i] = scores.second;
        // Early stopping if score == NaN
        if (scores.first != scores.first || scores.second != scores.second) {
            return scores.second;
        }
        mtrScore += scores.first;
        mtsScore += scores.second;
        if (verbose > 1) {
            std::cout << reccommend::now() << "train: " << scores.first << " - test: " << scores.second << "\n";
        }
    }
    mtrScore /= k;
    mtsScore /= k;
    // Compute score std
    for (int i = 0; i < k; i++) {
        strScore += (trainScores[i] - mtrScore) * (trainScores[i] - mtrScore);
        stsScore += (testScores[i] - mtsScore) * (testScores[i] - mtsScore);
    }
    strScore /= k;
    stsScore /= k;

    if (verbose > 0) {
        std::cout << reccommend::now() << k << "-fold CV. training score: "
                  << mtrScore << " +- " << strScore << " - "
                  << "test score: "
                  << mtsScore << " +- " << stsScore << ". "
                  << "Elapsed: " << reccommend::elapsed(start) << "\n";
    }

    return mtsScore;
}

// Explicit instantiation of the possible template functions.
// Whenever another solver is added its definition should also be
// appended here.
template double reccommend::kfoldCV<reccommend::BiasPredictor>(
                           const int k,
                           Settings &config,  
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::SVD>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::SimpleSGDSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::SGDppSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::IntegratedPearsonSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::IntegratedSpearmanSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::NeighbourhoodPearsonSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);

template double reccommend::kfoldCV<reccommend::NeighbourhoodSpearmanSolver>(
                           const int k,
                           Settings &config,
                           const std::vector< DataPair > &cvData,
                           const int verbose);


std::pair<double, double> reccommend::getScores(SGDSolver &solver) 
{
    auto predictors = solver.predictors();

    double tsScore = solver.testScore(predictors.second);
    double trScore = solver.trainScore(predictors.first);
    return std::pair<double, double>(trScore, tsScore);
}


