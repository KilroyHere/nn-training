#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nn {

namespace {

extern "C" {
void sgemv_(
    const char* trans,
    const int* m,
    const int* n,
    const float* alpha,
    const float* a,
    const int* lda,
    const float* x,
    const int* incx,
    const float* beta,
    float* y,
    const int* incy);
void saxpy_(const int* n, const float* alpha, const float* x, const int* incx, float* y, const int* incy);
}

BatchMetrics compute_metrics(const Matrix& probs, const std::vector<int>& y) {
    if (probs.rows != static_cast<int>(y.size())) {
        throw std::invalid_argument("compute_metrics dimension mismatch");
    }
    BatchMetrics m;
    int correct = 0;
    for (int i = 0; i < probs.rows; ++i) {
        const int label = y[static_cast<size_t>(i)];
        const float p = std::max(probs.at(i, label), 1e-8f);
        m.loss += -std::log(p);

        int pred = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < probs.cols; ++j) {
            if (probs.at(i, j) > best) {
                best = probs.at(i, j);
                pred = j;
            }
        }
        if (pred == label) {
            ++correct;
        }
    }

    m.loss /= static_cast<float>(probs.rows);
    m.accuracy = static_cast<float>(correct) / static_cast<float>(probs.rows);
    return m;
}

}  // namespace

MLP::MLP(const std::vector<int>& layer_sizes, std::mt19937& rng) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP requires at least input and output layers");
    }
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        const int in_dim = layer_sizes[i - 1];
        const int out_dim = layer_sizes[i];
        const float stddev = std::sqrt(2.0f / static_cast<float>(in_dim));

        Layer layer;
        layer.weights = random_normal(in_dim, out_dim, stddev, rng);
        layer.bias = std::vector<float>(static_cast<size_t>(out_dim), 0.0f);
        layers_.push_back(std::move(layer));
    }
}

BatchMetrics MLP::evaluate_batch(const Matrix& x, const std::vector<int>& y) const {
    Matrix a = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        Matrix z = matmul(a, layers_[i].weights);
        add_row_vector(&z, layers_[i].bias);
        if (i + 1 < layers_.size()) {
            relu_inplace(&z);
        }
        a = std::move(z);
    }
    const Matrix probs = softmax_rows(a);
    return compute_metrics(probs, y);
}

BatchMetrics MLP::train_batch(const Matrix& x, const std::vector<int>& y, float learning_rate) {
    std::vector<Matrix> activations;
    std::vector<Matrix> pre_activations;
    activations.reserve(layers_.size() + 1);
    pre_activations.reserve(layers_.size());
    activations.push_back(x);

    for (size_t i = 0; i < layers_.size(); ++i) {
        Matrix z = matmul(activations.back(), layers_[i].weights);
        add_row_vector(&z, layers_[i].bias);
        pre_activations.push_back(z);

        Matrix a = z;
        if (i + 1 < layers_.size()) {
            relu_inplace(&a);
        }
        activations.push_back(std::move(a));
    }

    Matrix probs = softmax_rows(activations.back());
    const BatchMetrics metrics = compute_metrics(probs, y);

    const float inv_batch = 1.0f / static_cast<float>(x.rows);
    for (int i = 0; i < probs.rows; ++i) {
        const int label = y[static_cast<size_t>(i)];
        probs.at(i, label) -= 1.0f;
    }
    for (float& v : probs.data) {
        v *= inv_batch;
    }

    Matrix grad = std::move(probs);
    for (int layer_idx = static_cast<int>(layers_.size()) - 1; layer_idx >= 0; --layer_idx) {
        const Matrix a_prev_t = transpose(activations[static_cast<size_t>(layer_idx)]);
        const Matrix grad_w = matmul(a_prev_t, grad);
        std::vector<float> grad_b(
            static_cast<size_t>(layers_[static_cast<size_t>(layer_idx)].bias.size()), 0.0f);
        std::vector<float> ones(static_cast<size_t>(grad.rows), 1.0f);
        const char trans_n = 'N';
        const int m = grad.cols;
        const int n = grad.rows;
        const int lda = grad.cols;
        const int inc_reduce = 1;
        const float alpha_reduce = 1.0f;
        const float beta_reduce = 0.0f;
        // grad is row-major [rows, cols], interpreted as col-major [cols, rows] = grad^T.
        sgemv_(
            &trans_n,
            &m,
            &n,
            &alpha_reduce,
            grad.data.data(),
            &lda,
            ones.data(),
            &inc_reduce,
            &beta_reduce,
            grad_b.data(),
            &inc_reduce);

        Matrix grad_prev;
        if (layer_idx > 0) {
            const Matrix wt = transpose(layers_[static_cast<size_t>(layer_idx)].weights);
            grad_prev = matmul(grad, wt);
            grad_prev = relu_backward(
                grad_prev, pre_activations[static_cast<size_t>(layer_idx - 1)]);
        }

        Layer& layer = layers_[static_cast<size_t>(layer_idx)];
        const int inc_update = 1;
        const int weight_size = static_cast<int>(layer.weights.data.size());
        const float alpha_update = -learning_rate;
        saxpy_(
            &weight_size,
            &alpha_update,
            grad_w.data.data(),
            &inc_update,
            layer.weights.data.data(),
            &inc_update);

        const int bias_size = static_cast<int>(layer.bias.size());
        saxpy_(
            &bias_size,
            &alpha_update,
            grad_b.data(),
            &inc_update,
            layer.bias.data(),
            &inc_update);

        if (layer_idx > 0) {
            grad = std::move(grad_prev);
        }
    }

    return metrics;
}

}  // namespace nn
