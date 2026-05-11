// MLP forward/backward implementation and BLAS-backed SGD updates.
#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nn {

namespace {

extern "C" {
void sgemm_(
    const char* transa,
    const char* transb,
    const int* m,
    const int* n,
    const int* k,
    const float* alpha,
    const float* a,
    const int* lda,
    const float* b,
    const int* ldb,
    const float* beta,
    float* c,
    const int* ldc);
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

void ensure_matrix_shape(Matrix* m, int rows, int cols) {
    if (m == nullptr) {
        throw std::invalid_argument("ensure_matrix_shape requires non-null matrix");
    }
    if (m->rows != rows || m->cols != cols) {
        m->rows = rows;
        m->cols = cols;
        m->data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    }
}

void matmul_into(const Matrix& a, const Matrix& b, Matrix* out) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("matmul dimension mismatch");
    }
    ensure_matrix_shape(out, a.rows, b.cols);
    const char trans_n = 'N';
    const int m = b.cols;
    const int n = a.rows;
    const int k = a.cols;
    const int lda = b.cols;
    const int ldb = a.cols;
    const int ldc = out->cols;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    sgemm_(
        &trans_n,
        &trans_n,
        &m,
        &n,
        &k,
        &alpha,
        b.data.data(),
        &lda,
        a.data.data(),
        &ldb,
        &beta,
        out->data.data(),
        &ldc);
}

void transpose_into(const Matrix& a, Matrix* out) {
    ensure_matrix_shape(out, a.cols, a.rows);
    for (int i = 0; i < a.rows; ++i) {
        const size_t src_base = static_cast<size_t>(i) * static_cast<size_t>(a.cols);
        for (int j = 0; j < a.cols; ++j) {
            out->data[static_cast<size_t>(j) * static_cast<size_t>(out->cols) + static_cast<size_t>(i)] =
                a.data[src_base + static_cast<size_t>(j)];
        }
    }
}

void relu_backward_inplace(Matrix* grad, const Matrix& pre_activation) {
    if (grad == nullptr) {
        throw std::invalid_argument("relu_backward_inplace requires non-null gradient");
    }
    if (grad->rows != pre_activation.rows || grad->cols != pre_activation.cols) {
        throw std::invalid_argument("relu_backward dimension mismatch");
    }
    const size_t n = grad->data.size();
    float* grad_ptr = grad->data.data();
    const float* pre_ptr = pre_activation.data.data();
    for (size_t i = 0; i < n; ++i) {
        grad_ptr[i] = pre_ptr[i] > 0.0f ? grad_ptr[i] : 0.0f;
    }
}

// Computes average cross-entropy loss and top-1 accuracy.
BatchMetrics compute_metrics(const Matrix& probs, const std::vector<int>& y) {
    if (probs.rows != static_cast<int>(y.size())) {
        throw std::invalid_argument("compute_metrics dimension mismatch");
    }
    BatchMetrics m;
    int correct = 0;
    for (int i = 0; i < probs.rows; ++i) {
        const float* row = probs.data.data() + static_cast<size_t>(i) * static_cast<size_t>(probs.cols);
        const int label = y[static_cast<size_t>(i)];
        const float p = std::max(row[label], 1e-8f);
        m.loss += -std::log(p);

        int pred = 0;
        float best = row[0];
        for (int j = 1; j < probs.cols; ++j) {
            if (row[j] > best) {
                best = row[j];
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

// Initializes layer weights and biases for the configured architecture.
MLP::MLP(const std::vector<int>& layer_sizes, std::mt19937& rng) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP requires at least input and output layers");
    }
    layer_sizes_ = layer_sizes;
    layers_.reserve(layer_sizes.size() - 1);
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

void MLP::ensure_workspace(int batch_rows) {
    if (batch_rows <= 0) {
        throw std::invalid_argument("batch size must be positive");
    }
    if (workspace_.batch_rows == batch_rows &&
        workspace_.activations.size() == layer_sizes_.size() &&
        workspace_.pre_activations.size() + 1 == layer_sizes_.size()) {
        return;
    }

    workspace_.batch_rows = batch_rows;
    workspace_.activations.resize(layer_sizes_.size());
    for (size_t i = 0; i < layer_sizes_.size(); ++i) {
        ensure_matrix_shape(&workspace_.activations[i], batch_rows, layer_sizes_[i]);
    }

    workspace_.pre_activations.resize(layer_sizes_.size() - 1);
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        ensure_matrix_shape(&workspace_.pre_activations[i], batch_rows, layer_sizes_[i + 1]);
    }

    ensure_matrix_shape(&workspace_.probs, batch_rows, layer_sizes_.back());
    ensure_matrix_shape(&workspace_.grad, batch_rows, layer_sizes_.back());
    ensure_matrix_shape(&workspace_.grad_prev, batch_rows, layer_sizes_.back());
    workspace_.ones.assign(static_cast<size_t>(batch_rows), 1.0f);
}

void MLP::ensure_gradient_buffer_shapes(const std::vector<Layer>& layers, GradientBuffers* gradients) {
    if (gradients == nullptr) {
        throw std::invalid_argument("gradients must be non-null");
    }
    gradients->weight_grads.resize(layers.size());
    gradients->bias_grads.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        ensure_matrix_shape(
            &gradients->weight_grads[i],
            layers[i].weights.rows,
            layers[i].weights.cols);
        gradients->bias_grads[i].assign(layers[i].bias.size(), 0.0f);
    }
}

// Runs inference-only forward pass and returns batch metrics.
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

// Runs forward/backward pass and stores gradients without applying updates.
BatchMetrics MLP::compute_batch_gradients(
    const Matrix& x,
    const std::vector<int>& y,
    GradientBuffers* gradients) {
    if (gradients == nullptr) {
        throw std::invalid_argument("compute_batch_gradients requires non-null gradients");
    }
    if (x.rows != static_cast<int>(y.size())) {
        throw std::invalid_argument("compute_batch_gradients x/y dimension mismatch");
    }
    if (x.cols != layer_sizes_.front()) {
        throw std::invalid_argument("compute_batch_gradients input feature dimension mismatch");
    }

    ensure_workspace(x.rows);
    ensure_gradient_buffer_shapes(layers_, gradients);

    Matrix& input_activation = workspace_.activations[0];
    std::copy(x.data.begin(), x.data.end(), input_activation.data.begin());
    for (size_t i = 0; i < layers_.size(); ++i) {
        Matrix& z = workspace_.pre_activations[i];
        matmul_into(workspace_.activations[i], layers_[i].weights, &z);
        add_row_vector(&z, layers_[i].bias);

        Matrix& a = workspace_.activations[i + 1];
        std::copy(z.data.begin(), z.data.end(), a.data.begin());
        if (i + 1 < layers_.size()) {
            relu_inplace(&a);
        }
    }

    const Matrix& logits = workspace_.activations.back();
    for (int i = 0; i < logits.rows; ++i) {
        const size_t row_base = static_cast<size_t>(i) * static_cast<size_t>(logits.cols);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < logits.cols; ++j) {
            max_logit = std::max(max_logit, logits.data[row_base + static_cast<size_t>(j)]);
        }
        float sum = 0.0f;
        for (int j = 0; j < logits.cols; ++j) {
            const float e = std::exp(logits.data[row_base + static_cast<size_t>(j)] - max_logit);
            workspace_.probs.data[row_base + static_cast<size_t>(j)] = e;
            sum += e;
        }
        const float inv = 1.0f / std::max(sum, 1e-12f);
        for (int j = 0; j < logits.cols; ++j) {
            workspace_.probs.data[row_base + static_cast<size_t>(j)] *= inv;
        }
    }
    const BatchMetrics metrics = compute_metrics(workspace_.probs, y);

    const float inv_batch = 1.0f / static_cast<float>(x.rows);
    for (int i = 0; i < workspace_.probs.rows; ++i) {
        const int label = y[static_cast<size_t>(i)];
        workspace_.probs.data[static_cast<size_t>(i) * static_cast<size_t>(workspace_.probs.cols) +
                             static_cast<size_t>(label)] -= 1.0f;
    }
    for (float& v : workspace_.probs.data) {
        v *= inv_batch;
    }

    workspace_.grad = workspace_.probs;
    const char trans_n = 'N';
    const int inc_reduce = 1;
    const float alpha_reduce = 1.0f;
    const float beta_reduce = 0.0f;
    for (int layer_idx = static_cast<int>(layers_.size()) - 1; layer_idx >= 0; --layer_idx) {
        transpose_into(workspace_.activations[static_cast<size_t>(layer_idx)], &workspace_.a_prev_t);
        matmul_into(
            workspace_.a_prev_t,
            workspace_.grad,
            &gradients->weight_grads[static_cast<size_t>(layer_idx)]);

        std::vector<float>& grad_b = gradients->bias_grads[static_cast<size_t>(layer_idx)];
        const int m = workspace_.grad.cols;
        const int n = workspace_.grad.rows;
        const int lda = workspace_.grad.cols;
        // grad is row-major [rows, cols], interpreted as col-major [cols, rows] = grad^T.
        sgemv_(
            &trans_n,
            &m,
            &n,
            &alpha_reduce,
            workspace_.grad.data.data(),
            &lda,
            workspace_.ones.data(),
            &inc_reduce,
            &beta_reduce,
            grad_b.data(),
            &inc_reduce);

        if (layer_idx > 0) {
            transpose_into(layers_[static_cast<size_t>(layer_idx)].weights, &workspace_.weight_t);
            matmul_into(workspace_.grad, workspace_.weight_t, &workspace_.grad_prev);
            relu_backward_inplace(
                &workspace_.grad_prev, workspace_.pre_activations[static_cast<size_t>(layer_idx - 1)]);
            std::swap(workspace_.grad, workspace_.grad_prev);
        }
    }

    return metrics;
}

// Applies precomputed gradients to model parameters with SGD.
void MLP::apply_gradients(const GradientBuffers& gradients, float learning_rate) {
    if (gradients.weight_grads.size() != layers_.size() ||
        gradients.bias_grads.size() != layers_.size()) {
        throw std::invalid_argument("Gradient buffer size mismatch in apply_gradients");
    }
    const int inc_update = 1;
    const float alpha_update = -learning_rate;
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        Layer& layer = layers_[layer_idx];
        const Matrix& grad_w = gradients.weight_grads[layer_idx];
        const std::vector<float>& grad_b = gradients.bias_grads[layer_idx];
        if (grad_w.rows != layer.weights.rows || grad_w.cols != layer.weights.cols) {
            throw std::invalid_argument("Weight gradient shape mismatch");
        }
        if (grad_b.size() != layer.bias.size()) {
            throw std::invalid_argument("Bias gradient shape mismatch");
        }
        const int weight_size = static_cast<int>(layer.weights.data.size());
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
    }
}

size_t MLP::weight_buffer_size() const {
    size_t total = 0;
    for (const Layer& l : layers_) {
        total += l.weights.data.size();
        total += l.bias.size();
    }
    return total;
}

void MLP::pack_weights(float* buf) const {
    size_t offset = 0;
    for (const Layer& l : layers_) {
        std::copy(l.weights.data.begin(), l.weights.data.end(), buf + offset);
        offset += l.weights.data.size();
        std::copy(l.bias.begin(), l.bias.end(), buf + offset);
        offset += l.bias.size();
    }
}

void MLP::unpack_weights(const float* buf) {
    size_t offset = 0;
    for (Layer& l : layers_) {
        std::copy(buf + offset, buf + offset + l.weights.data.size(), l.weights.data.begin());
        offset += l.weights.data.size();
        std::copy(buf + offset, buf + offset + l.bias.size(), l.bias.begin());
        offset += l.bias.size();
    }
}

// Convenience serial path: compute gradients and apply immediately.
BatchMetrics MLP::train_batch(const Matrix& x, const std::vector<int>& y, float learning_rate) {
    const BatchMetrics metrics = compute_batch_gradients(x, y, &train_gradients_);
    apply_gradients(train_gradients_, learning_rate);
    return metrics;
}

}  // namespace nn
