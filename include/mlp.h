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

    // Returns total number of weight + bias floats across all layers.
    size_t weight_buffer_size() const;
    // Copies all weights and biases into a caller-allocated flat buffer of size weight_buffer_size().
    void pack_weights(float* buf) const;
    // Overwrites all weights and biases from a flat buffer of size weight_buffer_size().
    void unpack_weights(const float* buf);

private:
    struct Workspace {
        int batch_rows = -1;
        std::vector<Matrix> activations;
        std::vector<Matrix> pre_activations;
        Matrix probs;
        Matrix grad;
        Matrix grad_prev;
        Matrix a_prev_t;
        Matrix weight_t;
        std::vector<float> ones;
    };

    void ensure_workspace(int batch_rows);
    static void ensure_gradient_buffer_shapes(const std::vector<Layer>& layers, GradientBuffers* gradients);

    std::vector<int> layer_sizes_;
    std::vector<Layer> layers_;
    Workspace workspace_;
    GradientBuffers train_gradients_;
};

}  // namespace nn

#endif  // NN_TRAINING_MLP_H_
