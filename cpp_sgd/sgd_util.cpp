#include <cmath>        // abs, sqrt

#include "sgd_util.h"
#include "sgd_types.h"

#include <Eigen/Dense>

using reccommend::dtype;
using reccommend::MatrixD;
using reccommend::MatrixI;
using reccommend::ColVectorD;
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
