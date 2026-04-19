#include "tensor.h"

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
void saxpy_(const int* n, const float* alpha, const float* x, const int* incx, float* y, const int* incy);
}

}  // namespace

Matrix::Matrix(int r, int c, float value) : rows(r), cols(c), data(r * c, value) {}

float& Matrix::at(int r, int c) {
    return data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
}

const float& Matrix::at(int r, int c) const {
    return data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)];
}

Matrix zeros(int rows, int cols) { return Matrix(rows, cols, 0.0f); }

Matrix random_normal(int rows, int cols, float stddev, std::mt19937& rng) {
    Matrix out(rows, cols, 0.0f);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (float& value : out.data) {
        value = dist(rng);
    }
    return out;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("matmul dimension mismatch");
    }
    Matrix out(a.rows, b.cols, 0.0f);
    const char trans_n = 'N';
    const int m = b.cols;
    const int n = a.rows;
    const int k = a.cols;
    const int lda = b.cols;
    const int ldb = a.cols;
    const int ldc = out.cols;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A * B mapped to column-major C^T = B^T * A^T.
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
        out.data.data(),
        &ldc);
    return out;
}

Matrix transpose(const Matrix& a) {
    Matrix out(a.cols, a.rows, 0.0f);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            out.at(j, i) = a.at(i, j);
        }
    }
    return out;
}

void add_row_vector(Matrix* a, const std::vector<float>& b) {
    if (a == nullptr || a->cols != static_cast<int>(b.size())) {
        throw std::invalid_argument("add_row_vector dimension mismatch");
    }
    const int n = a->cols;
    const int inc = 1;
    const float alpha = 1.0f;
    for (int i = 0; i < a->rows; ++i) {
        saxpy_(
            &n,
            &alpha,
            b.data(),
            &inc,
            &a->data[static_cast<size_t>(i) * static_cast<size_t>(a->cols)],
            &inc);
    }
}

void relu_inplace(Matrix* a) {
    if (a == nullptr) {
        throw std::invalid_argument("relu_inplace received null matrix");
    }
    for (float& v : a->data) {
        v = std::max(0.0f, v);
    }
}

Matrix relu_backward(const Matrix& grad, const Matrix& pre_activation) {
    if (grad.rows != pre_activation.rows || grad.cols != pre_activation.cols) {
        throw std::invalid_argument("relu_backward dimension mismatch");
    }
    Matrix out(grad.rows, grad.cols, 0.0f);
    for (size_t i = 0; i < grad.data.size(); ++i) {
        out.data[i] = pre_activation.data[i] > 0.0f ? grad.data[i] : 0.0f;
    }
    return out;
}

Matrix softmax_rows(const Matrix& logits) {
    Matrix out(logits.rows, logits.cols, 0.0f);
    for (int i = 0; i < logits.rows; ++i) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < logits.cols; ++j) {
            max_logit = std::max(max_logit, logits.at(i, j));
        }

        float sum = 0.0f;
        for (int j = 0; j < logits.cols; ++j) {
            const float shifted = logits.at(i, j) - max_logit;
            const float e = std::exp(shifted);
            out.at(i, j) = e;
            sum += e;
        }
        const float inv = 1.0f / std::max(sum, 1e-12f);
        for (int j = 0; j < logits.cols; ++j) {
            out.at(i, j) *= inv;
        }
    }
    return out;
}

}  // namespace nn
