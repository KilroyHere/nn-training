#ifndef NN_TRAINING_MLP_H_
#define NN_TRAINING_MLP_H_

#include <random>
#include <vector>

#include "tensor.h"

namespace nn {

struct BatchMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
};

struct GradientBuffers {
    std::vector<Matrix> weight_grads;
    std::vector<std::vector<float>> bias_grads;
};

struct Layer {
    Matrix weights;
    std::vector<float> bias;
};

class MLP {
public:
    MLP(const std::vector<int>& layer_sizes, std::mt19937& rng);

    BatchMetrics train_batch(const Matrix& x, const std::vector<int>& y, float learning_rate);
    BatchMetrics compute_batch_gradients(
        const Matrix& x,
        const std::vector<int>& y,
        GradientBuffers* gradients);
    void apply_gradients(const GradientBuffers& gradients, float learning_rate);
    BatchMetrics evaluate_batch(const Matrix& x, const std::vector<int>& y) const;

private:
    std::vector<Layer> layers_;
};

}  // namespace nn

#endif  // NN_TRAINING_MLP_H_
