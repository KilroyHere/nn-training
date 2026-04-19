#ifndef NN_TRAINING_CONFIG_H_
#define NN_TRAINING_CONFIG_H_

#include <string>
#include <vector>

namespace nn {

struct TrainConfig {
    int input_dim = 784;
    int num_classes = 10;
    std::vector<int> hidden_layers = {128, 64};

    int train_samples = 4096;
    int val_samples = 512;
    int batch_size = 64;
    int epochs = 5;
    float learning_rate = 0.02f;
    unsigned int seed = 42;

    std::string mnist_train_images = "data/mnist/train-images-idx3-ubyte";
    std::string mnist_train_labels = "data/mnist/train-labels-idx1-ubyte";
    std::string mnist_test_images = "data/mnist/t10k-images-idx3-ubyte";
    std::string mnist_test_labels = "data/mnist/t10k-labels-idx1-ubyte";

    std::string output_csv = "results/metrics.csv";
};

}  // namespace nn

#endif  // NN_TRAINING_CONFIG_H_
