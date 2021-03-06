/**
 * Generic implementation of stochastic gradient descent algorithm,
 * which is parallelized according to the TODO: Insert parallel algorithm
 * with OpenMP.
 * Along with the generic implementation which is in class SGDSolver this file contains
 * numerous classes which implement models for collaborative filtering, (mainly) solved using SGD.
 * Implemented algorithms:
 *  - BiasPredictor (non-SGD): calculate per-user and per-item bias values and use them for a 
 *      simple and not very accurate predictor.
 *  - SVD (non-SGD): performs a truncated singular value decomposition to obtain a predictor.
 *  - SimpleSGDSolver: automatically infer bias values via SGD, and obtain a predictor using such values.
 *  - SGDppSolver: implements the SVD++ model from TODO: insert paper. Integrates within the simple model
 *      implicit data obtained from the test set.
 *  - NeighbourhoodSolver: implements a neighbourhood model where we take into account the correlation between
 *      different feature vectors in the data. This does not come from an SVD solution like the other models.
 *  - IntegratedSolver: integrates the SVD++ and the neighbourhood models.
 *
 * author: gmeanti
 */

#ifndef __SGD_H
#define __SGD_H

#include <utility>      // pair
#include <vector>
#include <string>

#include <Eigen/Dense>

#include "sgd_types.h"

using namespace std;

namespace reccommend {
    /**
     * Base Stochastic gradient descent class.
     * Subclasses can use the SGD algorithm by implementing the predict, predictUpdate,
     * postIter methods.
     */
    class SGDSolver {
    protected:
        Settings m_settings;
        const MatrixI &m_train;
        const MatrixI &m_test;

        std::vector<pair<int, int> > m_trainIndices;
        std::vector<pair<int, int> > m_testIndices;     // Unused

        template <typename A>
        double score(const MatrixD &preds,
                     const Eigen::DenseBase<A> &truth)
        {
            double running = 0;
            int count = 0;
            for (int u = 0; u < truth.rows(); ++u) {
                for (int i = 0; i < truth.cols(); ++i) {
                    if (truth(u, i) > 0) {
                        running += pow(preds(u, i) - truth(u, i), 2);
                        count++;
                    }
                }
            }
            running /= count;
            return sqrt(running);
        }

        template <typename A>
        MatrixD singlePredictor(const Eigen::DenseBase<A> &data);
        
        std::vector<dtype> singlePredictor(const std::vector<pair<int, int> > &indices);

        virtual bool predictUpdate (int u, int i)=0;
        virtual bool initData ();
        virtual void postIter ()=0;
        virtual dtype predict (int u, int i)=0;
    public:
        SGDSolver(Settings settings, const MatrixI &train, const MatrixI &test)
            : m_train(train), m_test(test) {
                m_settings = settings;
        }

        dtype getSetting(string s) {
            return m_settings.at(s);
        }
        int getIntSetting(string s) {
            return static_cast<int>(m_settings.at(s));
        }
        void setSetting(string s, dtype v) {
            m_settings[s] = v;
        }

        SGDSolver& run();

        pair<MatrixD, MatrixD> predictors();
        
        double trainScore(const MatrixD &preds) {
            return score(preds, m_train);
        }

        double testScore(const MatrixD &preds) {
            return score(preds, m_test);
        }

        std::vector<dtype> trainPredictor () {
            return singlePredictor(m_trainIndices);
        }

        std::vector<dtype> testPredictor () {
            return singlePredictor(m_testIndices);
        }
    };

    /**
     * Non-SGD based predictor.
     * Calculates a "bias" value for each user-movie pair, based on the given training data.
     * Used parameters:
     * - K1
     * - K2
     */
    class BiasPredictor : public SGDSolver {
    public:
        using SGDSolver::SGDSolver;
        /* Simply return the u,i th item of the bias matrix */
        dtype predict (int u, int i) override;
    protected:
        // All work is done in initData
        bool initData () override;
        /* Empty function */
        bool predictUpdate (int u, int i) override;
        /* Empty function */
        void postIter () override;
    private:
        MatrixD m_biasMatrix;
        /* Holds the global mean of the training matrix. */
        dtype m_globalBias;
    };

    /**
     * Non-SGD based predictor.
     * Uses the biases from the BiasPredictor to fill in the missing values in the training data,
     * and then performs a truncated SVD.
     * Used parameters:
     * - K1
     * - K2
     * - num_factors
     */
    class SVD : public SGDSolver {
    public:
        using SGDSolver::SGDSolver;
        /* Simply return the u,i th item of the m_predictor matrix */
        dtype predict (int u, int i) override;
    protected:
        // All work is done in initData
        bool initData () override;
        /* Empty function */
        bool predictUpdate (int u, int i) override;
        /* Empty function */
        void postIter () override;
    private:
        /* Holds the result of the truncated SVD decomposition of the training matrix. */
        MatrixD m_predictor;
        /* Holds the global mean of the training matrix. */
        dtype m_globalBias;
    };

    /**
     * stochastic gradient descent solver for the SVD problem
     * with added user and item biases.
     * Used parameters:
     * - num_factors
     * - lrate1
     * - lrate2
     * - regl6
     * - regl7
     */
    class SimpleSGDSolver : public SGDSolver {
    public:
        using SGDSolver::SGDSolver;

        dtype predict (int u, int i) override;
    protected:
        bool initData () override;

        bool predictUpdate (int u, int i) override;

        void postIter () override;

        MatrixD m_userVecs;
        MatrixD m_itemVecs;
        ColVectorD m_userBias;
        ColVectorD m_itemBias;
        dtype m_globalBias;
    };


    /**
     * SVD++ model.
     * SGD solver for the SVD problem using additional "implicit"
     * information. Such information is taken from test data, and consists
     * of presence or absence of a vote (which is the only thing we can obtain
     * from the test set).
     * Used parameters:
     * - all of SimpleSGDSolver
     */
    class SGDppSolver : public SimpleSGDSolver
    {
    public:
        using SimpleSGDSolver::SimpleSGDSolver;

        dtype predict (int u, int i) override;
    protected:
        bool initData () override;

        bool predictUpdate (int u, int i) override;
        
        void postIter () override;

        MatrixD m_y; // Implicit information container
        vector<vector<int>> m_implicitU;
    };


    /**
     * Integrated model
     * Used parameters:
     * - num_factors
     * - K1, K2 (for bias)
     * - correlation_shrinkage (only when pearson corr is used)
     * - max_neigh
     * - lrate1, lrate2, lrate3
     * - regl6, regl7, regl8
     */
    class IntegratedSolver : public SGDppSolver
    {
    public:
        IntegratedSolver(Settings settings, const MatrixI &train, const MatrixI &test,
                         const std::string similarityType)
            : SGDppSolver(settings, train, test) {
            m_similarityType = similarityType;
        }

        dtype predict (int u, int i) override;
    protected:
        bool initData () override;
        bool predictUpdate (int u, int i) override;
        void postIter () override;

        std::string m_similarityType;
        MatrixD m_bias_offset;
        MatrixD m_simMat;
        MatrixD m_w;
        MatrixD m_c;
        vector<vector<int>> m_explicitU;
    };

    class IntegratedPearsonSolver : public IntegratedSolver {
    public:
        IntegratedPearsonSolver(Settings settings, const MatrixI &train, const MatrixI &test)
            : IntegratedSolver(settings, train, test, "pearson") {}
    };

    class IntegratedSpearmanSolver : public IntegratedSolver {
    public:
        IntegratedSpearmanSolver(Settings settings, const MatrixI &train, const MatrixI &test)
            : IntegratedSolver(settings, train, test, "spearman") {}
    };


    /**
     * Neighbourhood model
     * Used parameters:
     * - num_factors
     * - K1, K2 (for bias)
     * - correlation_shrinkage (only when pearson corr is used)
     * - max_neigh
     * - lrate1, lrate3
     * - regl4, regl6
     */
    class NeighbourhoodSolver : public IntegratedSolver
    {
    public:
        using IntegratedSolver::IntegratedSolver;

        dtype predict (int u, int i) override;
    protected:
        bool initData () override;

        bool predictUpdate (int u, int i) override;

        void postIter () override;
    };

    class NeighbourhoodPearsonSolver : public NeighbourhoodSolver {
    public:
        NeighbourhoodPearsonSolver(Settings settings, const MatrixI &train, const MatrixI &test)
            : NeighbourhoodSolver(settings, train, test, "pearson") {}
    };

    class NeighbourhoodSpearmanSolver : public NeighbourhoodSolver {
    public:
        NeighbourhoodSpearmanSolver(Settings settings, const MatrixI &train, const MatrixI &test)
            : NeighbourhoodSolver(settings, train, test, "spearman") {}
    };

}


#endif

