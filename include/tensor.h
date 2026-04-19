#ifndef NN_TRAINING_TENSOR_H_
#define NN_TRAINING_TENSOR_H_

#include <random>
#include <vector>

namespace nn {

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(int r, int c, float value = 0.0f);

    float& at(int r, int c);
    const float& at(int r, int c) const;
};

Matrix zeros(int rows, int cols);
Matrix random_normal(int rows, int cols, float stddev, std::mt19937& rng);
Matrix matmul(const Matrix& a, const Matrix& b);
Matrix transpose(const Matrix& a);
void add_row_vector(Matrix* a, const std::vector<float>& b);
void relu_inplace(Matrix* a);
Matrix relu_backward(const Matrix& grad, const Matrix& pre_activation);
Matrix softmax_rows(const Matrix& logits);

}  // namespace nn

#endif  // NN_TRAINING_TENSOR_H_
