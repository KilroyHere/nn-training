#ifndef NN_TRAINING_DATA_MNIST_H_
#define NN_TRAINING_DATA_MNIST_H_

#include "tensor.h"

namespace nn {

struct Dataset {
    Matrix features;
    std::vector<int> labels;
};

Dataset load_mnist_dataset(
    const std::string& image_path,
    const std::string& label_path,
    int max_samples = -1);
Dataset subset_dataset(const Dataset& full, const std::vector<int>& indices);

}  // namespace nn

#endif  // NN_TRAINING_DATA_MNIST_H_
