#include <cmath>         // abs, sqrt
#include <algorithm>     // min, set_intersection

#include <Eigen/Dense>

#include "sgd_util.h"
#include "sgd_types.h"


using reccommend::dtype;
using reccommend::MatrixD;
using reccommend::MatrixI;
using reccommend::ColVectorD;
using reccommend::ColVectorI;
using reccommend::RowVectorD;


dtype _calcPearson(const int i1, const int i2, 
                   const int lambda2, const MatrixI &data) 
{
    // Use formula (single pass) from
    // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Definition
    int inter_len = 0;
    int sum_prod = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sumsq1 = 0;
    int sumsq2 = 0;

    for (int u = 0; u < data.rows(); u++) {
        if (data(u, i1) > 0 && data(u, i2) > 0) {
            inter_len++;
            sum1 += data(u, i1);
            sumsq1 += data(u, i1)*data(u, i1);
            sum_prod += data(u, i1) * data(u, i2);
            sum2 += data(u, i2);
            sumsq2 += data(u, i2)*data(u, i2);
        }
    }

    long num = inter_len*sum_prod - sum1*sum2;
    double den = sqrt((inter_len*sumsq1 - sum1*sum1) * 
                      (inter_len*sumsq2 - sum2*sum2));
    // num / den is the Pearson correlation coefficient.
    // According to <paper> we use the shrunk correlation coefficient.
    // http://cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf
    double shrunk_c = static_cast<double>(inter_len) / (inter_len + lambda2) * abs(num/den);
    return static_cast<dtype>(shrunk_c);
}


ColVectorD _rank(const ColVectorI &v) {
    // Use vector because it's easier to sort than the Eigen data-structures
    std::vector<std::size_t> w(v.size());
    std::iota(begin(w), end(w), 0);

    std::sort(w.begin(), w.end(), [&v](std::size_t i, std::size_t j) { return v[i] < v[j]; });

    ColVectorD r(w.size());

    for (std::size_t n, i = 0; i < w.size(); i += n)
    {
        n = 1;
        while (i + n < w.size() && v[w[i]] == v[w[i+n]]) ++n;
        for (std::size_t k = 0; k < n; ++k)
        {
            r[w[i+k]] = i + (n + 1) / 2.0; // average rank of n tied values
        }
    }
    return r;
}

dtype _calcSpearman(const int i1, const int i2, const MatrixI &data) {
    // Need to find indices of users which have rated both items (`nonzero`).
    std::vector<int> nz1;
    std::vector<int> nz2;
    for (int u = 0; u < data.rows(); u++) {
        if (data(u, i1) != 0) nz1.emplace_back(u);
        if (data(u, i2) != 0) nz2.emplace_back(u);
    }
    std::vector<int> nonzero = std::vector<int>(std::min(nz1.size(), nz2.size()));
    auto it = std::set_intersection (nz1.begin(), nz1.end(), nz2.begin(), nz2.end(), nonzero.begin());
    nonzero.resize(it - nonzero.begin());

    ColVectorI v1(nonzero.size());
    ColVectorI v2(nonzero.size());

    for (std::size_t i = 0; i < nonzero.size(); i++) {
        v1[i] = data(nonzero[i], i1);
        v2[i] = data(nonzero[i], i2);
    }

    ColVectorD r1 = _rank(v1);
    ColVectorD r2 = _rank(v2);

    // Finally we have the ranked vectors r1 and r2
    // Now we apply formula from https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    int L = r1.size();
    dtype sum = 0;
    for (int l = 0; l < L; l++) {
        sum += (r1[l] - r2[l]) * (r1[l] - r2[l]);
    }
    return 1 - 6*sum / (L*L*L - L);
}


MatrixD reccommend::calcSpearmanMatrix (const MatrixI &data, const int num_threads) {
    int nitems = data.cols();

    MatrixD corrmat = MatrixD::Zero(nitems, nitems);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nitems; i++) {
            for (int j = i+1; j < nitems; j++) {
                corrmat(i, j) = _calcSpearman(i, j, data);
            }
        }
    }
    corrmat = corrmat + corrmat.adjoint().eval();
    return corrmat;
}


MatrixD reccommend::calcPearsonMatrix(const MatrixI &data, const int shrinkage, const int num_threads) {
    int nitems = data.cols();

    MatrixD corrmat = MatrixD::Zero(nitems, nitems);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nitems; i++) {
            for (int j = i+1; j < nitems; j++) {
                corrmat(i, j) = _calcPearson(i, j, shrinkage, data);
            }
        }
    }
    corrmat = corrmat + corrmat.adjoint().eval();
    return corrmat;
}



dtype reccommend::calcGlobalMean (const MatrixI &data, const std::vector<std::pair<int, int> > &dataIndices) {
    dtype mean = 0;
    for (auto it = dataIndices.begin(); it != dataIndices.end(); ++it) {
        mean += data(it->first, it->second);
    }
    mean /= dataIndices.size();
    return mean;
}



MatrixD reccommend::calcBiasMatrix (const MatrixI &data, const dtype globalMean, const int K1, const int K2) {
    int nitems = data.cols();
    int nusers = data.rows();

    MatrixD biasMatrix = MatrixD::Zero(nusers, nitems);
    ColVectorD movieMeans = ColVectorD::Zero(nitems);
    ColVectorD userOffsets = ColVectorD::Zero(nusers);

    // Calculate a (normalized) mean of each movie
    for (int i = 0; i < nitems; i++) {
        int sum = 0, count = 0;
        for (int u = 0; u < nusers; u++) {
            if (data(u, i) != 0) {
                sum += data(u, i);
                count++;
            }
        }
        movieMeans(i) = static_cast<dtype>(sum - globalMean*count) / (K1 + count);
    }

    // Calculate the "offset" for each user. Intuitively the offset represents
    // the tendency of a user to give higher or lower votes.
    for (int u = 0; u < nusers; u++) {
        dtype offsetSum = 0;
        int count = 0;
        for (int i = 0; i < nitems; i++) {
            if (data(u, i) != 0) {
                offsetSum += data(u, i) - movieMeans(i);
                count++;
            }
        }
        userOffsets(u) = (offsetSum - globalMean*count) / (K2 + count);
    }

    for (int u = 0; u < nusers; u++) {
        for (int i = 0; i < nitems; i++) {
            if (data(u, i) == 0) {
                biasMatrix(u, i) = movieMeans(i) + userOffsets(u) + globalMean;
            }
            else {
                biasMatrix(u, i) = data(u, i);
            }
        }
    }
    return biasMatrix;
}
