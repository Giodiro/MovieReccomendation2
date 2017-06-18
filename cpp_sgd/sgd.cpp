/**
 * Implementation file for sgd.h
 *
 * author: gmeanti
 */
#include <iostream>
#include <cmath>        // pow, sqrt, abs
#include <random>
#include <chrono>
#include <functional>   // bind

#include <Eigen/Dense>

#include "sgd.h"
#include "ioutil.h"     // for timing functions
#include "sgd_util.h"   // for computing various statistics (e.g. mean, bias) on the data matrices

//#define PARALLEL_REPORT
#define CHECK_NAN

using reccommend::SGDSolver;

using reccommend::dtype;
using reccommend::MatrixD;
using reccommend::MatrixI;
using reccommend::ColVectorD;
using reccommend::RowVectorD;


inline float gaussianSample(float mean, float stdev, dtype dummy) {
  static mt19937 rng;
  static normal_distribution<float> nd(mean, stdev);
  return nd(rng);
}

/**
 * SGDSolver (base class)
 */

bool reccommend::SGDSolver::initData () {
    for (int i = 0; i < m_train.rows(); ++i) {
        for (int j = 0; j < m_train.cols(); ++j) {
            if (m_train(i, j) > 0) {
                m_trainIndices.push_back(std::pair <int, int> (i, j));
            }
            if (m_test(i, j) > 0) {
                m_testIndices.push_back(std::pair <int, int> (i, j));
            }
        }
    }
    return true;
}

std::vector<dtype> reccommend::SGDSolver::singlePredictor(const std::vector<pair<int, int> > &indices) {
    std::vector<dtype> preds(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        preds[i] = predict(indices[i].first, indices[i].second);
    }
    return preds;
}

template <typename A>
MatrixD reccommend::SGDSolver::singlePredictor(const Eigen::DenseBase<A> &data) {
    // Iterate through all non-zero entries in the data matrix
    // and perform predictions in those points.
    MatrixD predMat(data.rows(), data.cols());

    // TODO: Could be sped up using train_indices directly
    // But then must also calculate test indices!
    for (int u = 0; u < data.rows(); ++u) {
        for (int i = 0; i < data.cols(); ++i) {
            if (data(u, i) > 0) {
                predMat(u, i) = predict(u, i);
            }
        }
    }

    return predMat;
}

std::pair<MatrixD, MatrixD> reccommend::SGDSolver::predictors () {
    // Training predictor
    MatrixD trainPred = singlePredictor(m_train);
    // Test predictor
    MatrixD testPred = singlePredictor(m_test);

    return std::pair<MatrixD, MatrixD> (trainPred, testPred);
}


SGDSolver& reccommend::SGDSolver::run () {
    volatile int  update      = 1; /* Thread shared variable holding the number of updates made so far */
    volatile bool die_flag    = false; /* Thread shared variable indicating whether all threads should stop */

    /**
       Initialize the necessary data structures for predictions.
       If initialization returns false then we won't proceed with updates.
       This mechanism is used with the SVD class where all work is performed 
       during the initialization procedure.
    */
    if (!initData()) {
        die_flag = true;
    }

    const    uint max_updates = getIntSetting("max_iter") * m_trainIndices.size();
    const    int  mod_iter    = max_updates / getIntSetting("max_iter");

    const    int  nt          = getIntSetting("num_threads");
    const    auto start       = chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(nt)
    {
        // Random number generation
        std::mt19937 rng(static_cast<uint>(omp_get_thread_num() + 1));
        std::uniform_int_distribution<int> uni(0, m_trainIndices.size() - 1);
        int curUpdate; /* Thread local variable to store the current update */

        // This is the loop that should be performed in parallel!
        #pragma omp for schedule(static)
        for (uint k = 1; k <= max_updates; k++) {
            #pragma omp flush (die_flag)
            if (!die_flag) {
                #pragma omp atomic read
                curUpdate = update;
                if (curUpdate % mod_iter == 0) {
                    #ifdef PARALLEL_REPORT
                        cout << reccommend::now() << "Update " << curUpdate << " - "
                             << "Full iteration " << (curUpdate / mod_iter)
                             << " finished. (" << static_cast<int>(static_cast<float>(curUpdate) / max_updates * 100)
                             << "%) - took " << reccommend::elapsed(start) << "ms\n";
                        MatrixD testPredictor = singlePredictor(m_test);
                        cout << "lrate1: " << getSetting("lrate1") << " - Test score: " << testScore(testPredictor) << "\n";
                    #endif
                    postIter();
                }
                pair<int, int> loc = m_trainIndices[static_cast<uint>(uni(rng))];
                if (!predictUpdate(loc.first, loc.second)) {
                    die_flag = true;
                    #pragma omp flush (die_flag)
                }

                #pragma omp atomic update
                update += 1;
            }
        }
    }
    return *this;
}


/**
 * Bias Predictor
 */

bool reccommend::BiasPredictor::initData () {
    SGDSolver::initData();
    m_globalBias = reccommend::calcGlobalMean(m_train, m_trainIndices);
    m_biasMatrix = reccommend::calcBiasMatrix(m_train, m_globalBias, getIntSetting("K1"), getIntSetting("K2"));
    // Return false since we don't want to perform SGD!
    return false;
}

bool reccommend::BiasPredictor::predictUpdate(int u, int i) {
    return false;
}

dtype reccommend::BiasPredictor::predict(int u, int i) {
    return m_biasMatrix(u, i);
}

void reccommend::BiasPredictor::postIter() {}


/**
 * SVD
 */

bool reccommend::SVD::initData () {
    SGDSolver::initData();

    m_globalBias = reccommend::calcGlobalMean(m_train, m_trainIndices);
    MatrixD biasMatrix = reccommend::calcBiasMatrix(m_train, m_globalBias, getIntSetting("K1"), getIntSetting("K2"));

    // Perform SVD
    // TODO: directly calculate truncated SVD (apparently not possible with Eigen)
    auto decomp = Eigen::BDCSVD<MatrixD>(biasMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Truncate the decomposition up to "num_factors" principal directions
    int trunc = getIntSetting("num_factors");
    auto U = decomp.matrixU().leftCols(trunc);
    auto D = decomp.singularValues().head(trunc).asDiagonal();
    auto V = decomp.matrixV().leftCols(trunc);

    // TODO: Prevent aliasing??
    m_predictor.noalias() = U * D * V.transpose();

    // Return false since we don't want to perform SGD!
    return false;
}


bool reccommend::SVD::predictUpdate(int u, int i) {
    return false;
}

dtype reccommend::SVD::predict(int u, int i) {
    return m_predictor(u, i);
}

void reccommend::SVD::postIter() {}


/**
 * SimpleSGDSolver (basic model)
 */

dtype reccommend::SimpleSGDSolver::predict (int u, int i) {
    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                 m_userVecs.row(u) * m_itemVecs.row(i).adjoint();
    return prediction;
}

bool reccommend::SimpleSGDSolver::initData() {
    SGDSolver::initData();

    float stdev = 1 / getSetting("num_factors");
    std::function<float(float)> stdSample = std::bind(gaussianSample, 0.0f, stdev, std::placeholders::_1);

    m_userVecs = MatrixD(getIntSetting("nusers"), getIntSetting("num_factors")).unaryExpr(stdSample);
    m_itemVecs = MatrixD(getIntSetting("nitems"), getIntSetting("num_factors")).unaryExpr(stdSample);
    m_userBias = ColVectorD::Zero(getIntSetting("nusers"));
    m_itemBias = ColVectorD::Zero(getIntSetting("nitems"));
    

    // Global bias is the mean of all non-zero values.
    // This could be more efficient if train were a SparseMatrix??
    // http://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    m_globalBias = reccommend::calcGlobalMean(m_train, m_trainIndices);

    return true;
}

bool reccommend::SimpleSGDSolver::predictUpdate (int u, int i) {
    dtype p = predict(u, i);
    dtype err = m_train(u, i) - p;
    m_userBias(u) += getSetting("lrate1") * (err - getSetting("regl6") * m_userBias(u));
    m_itemBias(i) += getSetting("lrate1") * (err - getSetting("regl6") * m_itemBias(i));
    // To perform componentwise subtraction maybe need to use .array() to change view
    m_userVecs.row(u) += getSetting("lrate2") * 
        (err * m_itemVecs.row(i) - getSetting("regl7") * m_userVecs.row(u));
    m_itemVecs.row(i) += getSetting("lrate2") * 
        (err * m_userVecs.row(u) - getSetting("regl7") * m_itemVecs.row(i));
    return true;
}

void reccommend::SimpleSGDSolver::postIter () {
    dtype lrate1 = getSetting("lrate1");
    dtype lrate2 = getSetting("lrate2");
    setSetting("lrate1", lrate1 * getSetting("lrate_reduction"));
    setSetting("lrate2", lrate2 * getSetting("lrate_reduction"));
}



/**
 * SGDppSolver (SVD++). Inherits from the basic model
 */

bool reccommend::SGDppSolver::initData () {
    // Initialize from super class
    SimpleSGDSolver::initData();

    m_y = MatrixD::Zero(getIntSetting("nitems"), getIntSetting("num_factors"));

    // Precompute the which_num_u arrays (indices of ratings sorted by users)
    m_implicitU = std::vector<vector<int> >(getIntSetting("nusers"));
    for (uint u = 0; u < m_train.rows(); ++u) {
        vector<int> imp;
        for (uint i = 0; i < m_train.cols(); ++i) {
            if (m_train(u, i) > 0 || m_test(u, i) > 0) {
                imp.push_back(i);
            }
        }
        m_implicitU[u] = imp;
    }
    return true;
}

dtype reccommend::SGDppSolver::predict (int u, int i) {
    std::vector<int> implicitU = m_implicitU[u];
    dtype n_impl_tot = 1 / sqrt(implicitU.size());
    
    RowVectorD sum_y = RowVectorD::Zero(getIntSetting("num_factors"));   
    for (size_t a = 0; a < implicitU.size(); a++) {
        sum_y += m_y.row(implicitU[a]);
    }
    // Here we already integrate the userVecs within sum_y
    sum_y = n_impl_tot * sum_y + m_userVecs.row(u);

    /* PREDICT */
    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                       m_itemVecs.row(i) * sum_y.adjoint();
    return prediction;
}

bool reccommend::SGDppSolver::predictUpdate (int u, int i) {
    // See second half of page 6 in
    // http://cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf
    // (this is the SVD++ model)
    std::vector<int> implicitU = m_implicitU[u];
    dtype n_impl_tot = 1 / sqrt(implicitU.size());
    
    RowVectorD sum_y = RowVectorD::Zero(getIntSetting("num_factors"));   
    for (size_t a = 0; a < implicitU.size(); a++) {
        sum_y += m_y.row(implicitU[a]);
    }
    // Here we already integrate the userVecs within sum_y
    sum_y = n_impl_tot * sum_y + m_userVecs.row(u);

    /* PREDICT */
    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                       m_itemVecs.row(i) * sum_y.adjoint();

    /* UPDATE */
    dtype err = m_train(u, i) - prediction;

    m_userBias(u) += getSetting("lrate1") * (err - getSetting("regl6") * m_userBias(u));
    m_itemBias(i) += getSetting("lrate1") * (err - getSetting("regl6") * m_itemBias(i)); 

    m_userVecs.row(u) += getSetting("lrate2") * 
        (err * m_itemVecs.row(i) - getSetting("regl7") * m_userVecs.row(u));
    m_itemVecs.row(i) += getSetting("lrate2") * 
        (err * sum_y - getSetting("regl7") * m_itemVecs.row(i));
    
    RowVectorD y_update_prec = getSetting("lrate2") * err * n_impl_tot * m_itemVecs.row(i);
    double l2r7 = getSetting("lrate2") * getSetting("regl7");
    for (size_t a = 0; a < implicitU.size(); a++) {
        m_y.row(implicitU[a]) += y_update_prec -
            l2r7 * m_y.row(implicitU[a]);
    }
    return true;
}

void reccommend::SGDppSolver::postIter (){
    SimpleSGDSolver::postIter();
}

/**
 * Integrated model
 */

bool reccommend::IntegratedSolver::initData () {
    // Initialize from super class
    SGDppSolver::initData();

    m_w = MatrixD::Zero(getIntSetting("nitems"), getIntSetting("nitems"));
    m_c = MatrixD::Zero(getIntSetting("nitems"), getIntSetting("nitems"));
    m_bias_offset = m_train.cast<dtype>() - 
                    reccommend::calcBiasMatrix(m_train, m_globalBias, 
                                               getIntSetting("K1"), getIntSetting("K2"));

    // Similarity score depends on constructor parameter `m_similarityType`:
    if (m_similarityType == "pearson") {
        m_simMat = reccommend::calcPearsonMatrix(m_train, getIntSetting("correlation_shrinkage"),
                                                   getIntSetting("num_threads"));
    } else if (m_similarityType == "spearman") {
        m_simMat = reccommend::calcSpearmanMatrix(m_train, getIntSetting("num_threads"));
    } else {
        std::cout << "Invalid similarity score " << m_similarityType << "\n";
        return false;
    }

    // Precompute the which_num_u arrays (indices of ratings sorted by users)
    m_explicitU = std::vector<vector<int> >(getIntSetting("nusers"));
    for (int u = 0; u < m_train.rows(); ++u) {
        vector<int> exp;
        for (int i = 0; i < m_train.cols(); ++i) {
            if (m_train(u, i) > 0) {
                exp.push_back(i);
            }
        }
        m_explicitU[u] = exp;
    }
    return true;
}

dtype reccommend::IntegratedSolver::predict (int u, int i) {
    // Copy the explicit/implicit user rating indices vectors into local copies
    std::vector<int> explicitU(m_explicitU[u].size());
    std::copy(m_explicitU[u].begin(), m_explicitU[u].end(), explicitU.begin());
    std::vector<int> implicitU(m_implicitU[u].size());
    std::copy(m_implicitU[u].begin(), m_implicitU[u].end(), implicitU.begin());

    int inv_kmax_expl = max(static_cast<dtype>(0), explicitU.size() - getSetting("max_neigh"));
    int inv_kmax_impl = max(static_cast<dtype>(0), implicitU.size() - getSetting("max_neigh"));
    // partial sort of the implicitU, explicitU vectors so that all elements
    // after position inv_kmax_* are _more similar_ to the current item i, than
    // the elements before that position.
    nth_element(explicitU.begin(), 
                explicitU.begin() + inv_kmax_expl,
                explicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });
    nth_element(implicitU.begin(), 
                implicitU.begin() + inv_kmax_impl,
                implicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });

    dtype n_expl = 1 / sqrt(explicitU.size() - inv_kmax_expl);
    dtype n_impl = 1 / sqrt(implicitU.size() - inv_kmax_impl);
    dtype n_impl_tot = 1 / sqrt(implicitU.size());

    // Start calculating the components of prediction:
    // 1) y (in this instance we sum over y for all items in implicitU)
    RowVectorD sum_y = RowVectorD::Zero(getIntSetting("num_factors"));   
    for (size_t a = 0; a < implicitU.size(); a++) {
        sum_y += m_y.row(implicitU[a]);
    }
    // Here we already integrate the userVecs within sum_y
    sum_y = n_impl_tot * sum_y + m_userVecs.row(u);

    // 2) Neighbourhood model (explicit)
    dtype neighbour_sum = 0;
    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        neighbour_sum += m_w(i, explicitU[a]) * 
            (m_bias_offset(u, explicitU[a]));
    }

    // 3) Neighbourhood model (implicit)
    dtype implicit_sum = 0;
    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        implicit_sum += m_c(i, implicitU[a]);
    }

    /* PREDICT */
    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                        m_itemVecs.row(i) * sum_y.adjoint() +
                        n_expl * neighbour_sum +
                        n_impl * implicit_sum;
    return prediction;
}


bool reccommend::IntegratedSolver::predictUpdate (int u, int i) {
    /*
     * Optimization candidates:
     *  - precompute m_train - m_fixed_bias (this is also the only place where m_fixed_bias appears)
     *  - 
     *
     */
    // Copy the explicit/implicit user rating indices vectors into local copies
    std::vector<int> explicitU(m_explicitU[u].size());
    std::copy(m_explicitU[u].begin(), m_explicitU[u].end(), explicitU.begin());
    std::vector<int> implicitU(m_implicitU[u].size());
    std::copy(m_implicitU[u].begin(), m_implicitU[u].end(), implicitU.begin());

    int inv_kmax_expl = max(static_cast<dtype>(0), explicitU.size() - getSetting("max_neigh"));
    int inv_kmax_impl = max(static_cast<dtype>(0), implicitU.size() - getSetting("max_neigh"));
    // partial sort of the implicitU, explicitU vectors so that all elements
    // after position inv_kmax_* are _more similar_ to the current item i, than
    // the elements before that position.
    nth_element(explicitU.begin(), 
                explicitU.begin() + inv_kmax_expl,
                explicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });
    nth_element(implicitU.begin(), 
                implicitU.begin() + inv_kmax_impl,
                implicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });

    dtype n_expl = 1 / sqrt(explicitU.size() - inv_kmax_expl);
    dtype n_impl = 1 / sqrt(implicitU.size() - inv_kmax_impl);
    dtype n_impl_tot = 1 / sqrt(implicitU.size());

    // Start calculating the components of prediction:
    // 1) y (in this instance we sum over y for all items in implicitU)
    RowVectorD sum_y = RowVectorD::Zero(getIntSetting("num_factors"));   
    for (size_t a = 0; a < implicitU.size(); a++) {
        sum_y += m_y.row(implicitU[a]);
    }
    // Here we already integrate the userVecs within sum_y
    sum_y = n_impl_tot * sum_y + m_userVecs.row(u);

    // 2) Neighbourhood model (explicit)
    dtype neighbour_sum = 0;
    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        neighbour_sum += m_w(i, explicitU[a]) * m_bias_offset(u, explicitU[a]);
    }

    // 3) Neighbourhood model (implicit)
    dtype implicit_sum = 0;
    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        implicit_sum += m_c(i, implicitU[a]);
    }

    /* PREDICT */
    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                        m_itemVecs.row(i) * sum_y.adjoint() +
                        n_expl * neighbour_sum +
                        n_impl * implicit_sum;
    #ifdef CHECK_NAN
        if (prediction != prediction) {
            return false;
        }
    #endif

    /* UPDATE */
    dtype err = m_train(u, i) - prediction;

    m_userBias(u) += getSetting("lrate1") * (err - getSetting("regl6") * m_userBias(u));
    m_itemBias(i) += getSetting("lrate1") * (err - getSetting("regl6") * m_itemBias(i)); 

    m_userVecs.row(u) += getSetting("lrate2") * 
        (err * m_itemVecs.row(i) - getSetting("regl7") * m_userVecs.row(u));
    m_itemVecs.row(i) += getSetting("lrate2") * 
        (err * sum_y - getSetting("regl7") * m_itemVecs.row(i));
    
    RowVectorD y_update_prec = getSetting("lrate2") * err * n_impl_tot * m_itemVecs.row(i);
    double l2r7 = getSetting("lrate2") * getSetting("regl7");
    for (size_t a = 0; a < implicitU.size(); a++) {
        m_y.row(implicitU[a]) += y_update_prec - l2r7 * m_y.row(implicitU[a]);
    }

    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        m_w(i, explicitU[a]) += getSetting("lrate3") *
            (err * n_expl * m_bias_offset(u, explicitU[a]) - getSetting("regl8") * m_w(i, explicitU[a]));
    }

    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        m_c(i, implicitU[a]) += getSetting("lrate3") *
            (err * n_impl - getSetting("regl8") * m_c(i, implicitU[a]));
    }
    return true;
}

void reccommend::IntegratedSolver::postIter () {
    SGDppSolver::postIter();
    dtype lrate3 = getSetting("lrate3");
    setSetting("lrate3", lrate3 * getSetting("lrate_reduction"));
}



/**
 * Neighbourhood Model
 */

bool reccommend::NeighbourhoodSolver::initData () {
    return IntegratedSolver::initData();
}

dtype reccommend::NeighbourhoodSolver::predict (int u, int i) {
    // Copy the explicit/implicit user rating indices vectors into local copies
    std::vector<int> explicitU(m_explicitU[u].size());
    std::copy(m_explicitU[u].begin(), m_explicitU[u].end(), explicitU.begin());
    std::vector<int> implicitU(m_implicitU[u].size());
    std::copy(m_implicitU[u].begin(), m_implicitU[u].end(), implicitU.begin());

    int inv_kmax_expl = max(static_cast<dtype>(0), explicitU.size() - getSetting("max_neigh"));
    int inv_kmax_impl = max(static_cast<dtype>(0), implicitU.size() - getSetting("max_neigh"));

    nth_element(explicitU.begin(), 
                explicitU.begin() + inv_kmax_expl,
                explicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });
    nth_element(implicitU.begin(), 
                implicitU.begin() + inv_kmax_impl,
                implicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });

    dtype n_expl = 1 / sqrt(explicitU.size() - inv_kmax_expl);
    dtype n_impl = 1 / sqrt(implicitU.size() - inv_kmax_impl);

    //cout << n_expl << " " << n_impl << std::endl;

    dtype explicit_sum = 0;
    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        explicit_sum += m_bias_offset(u, explicitU[a]) * m_w(i, explicitU[a]);
    }
    dtype implicit_sum = 0;
    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        implicit_sum += m_c(i, implicitU[a]);
    }
    //cout << explicit_sum << " " << implicit_sum << std::endl;


    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                    n_expl * explicit_sum +
                    n_impl * implicit_sum;
    return prediction;
}

bool reccommend::NeighbourhoodSolver::predictUpdate (int u, int i) {
    // Copy the explicit/implicit user rating indices vectors into local copies
    std::vector<int> explicitU(m_explicitU[u].size());
    std::copy(m_explicitU[u].begin(), m_explicitU[u].end(), explicitU.begin());
    std::vector<int> implicitU(m_implicitU[u].size());
    std::copy(m_implicitU[u].begin(), m_implicitU[u].end(), implicitU.begin());

    int inv_kmax_expl = max(static_cast<dtype>(0), explicitU.size() - getSetting("max_neigh"));
    int inv_kmax_impl = max(static_cast<dtype>(0), implicitU.size() - getSetting("max_neigh"));

    nth_element(explicitU.begin(), 
                explicitU.begin() + inv_kmax_expl,
                explicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });
    nth_element(implicitU.begin(), 
                implicitU.begin() + inv_kmax_impl,
                implicitU.end(),
                [this, i] (const int i1, const int i2) {
                    return m_simMat(i, i1) < m_simMat(i, i2);
                });

    dtype n_expl = 1 / sqrt(explicitU.size() - inv_kmax_expl);
    dtype n_impl = 1 / sqrt(implicitU.size() - inv_kmax_impl);

    dtype explicit_sum = 0;
    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        explicit_sum += m_bias_offset(u, explicitU[a]) * m_w(i, explicitU[a]);
    }
    dtype implicit_sum = 0;
    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        implicit_sum += m_c(i, implicitU[a]);
    }

    dtype prediction = m_globalBias + m_userBias(u) + m_itemBias(i) +
                    n_expl * explicit_sum +
                    n_impl * implicit_sum;

    // UPDATE
    dtype err = m_train(u, i) - prediction;

    m_userBias(u) += getSetting("lrate1") * (err - getSetting("regl6") * m_userBias(u));
    m_itemBias(i) += getSetting("lrate1") * (err - getSetting("regl6") * m_itemBias(i)); 

    // Update m_w
    for (size_t a = inv_kmax_expl; a < explicitU.size(); a++) {
        m_w(i, explicitU[a]) += getSetting("lrate3") *
            (err * n_expl * m_bias_offset(u, explicitU[a]) - getSetting("regl4") * m_w(i, explicitU[a]));
    }
    // Update m_c
    for (size_t a = inv_kmax_impl; a < implicitU.size(); a++) {
        m_c(i, implicitU[a]) += getSetting("lrate3") *
            (err * n_impl - getSetting("regl4") * m_c(i, implicitU[a]));
    }
    return true;
}


void reccommend::NeighbourhoodSolver::postIter () {
    IntegratedSolver::postIter();
}
