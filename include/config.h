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
    int microbatch_count = 4;
    int epochs = 5;
    float learning_rate = 0.02f;
    unsigned int seed = 42;

    std::string mnist_train_images = "data/mnist/train-images-idx3-ubyte";
    std::string mnist_train_labels = "data/mnist/train-labels-idx1-ubyte";
    std::string mnist_test_images = "data/mnist/t10k-images-idx3-ubyte";
    std::string mnist_test_labels = "data/mnist/t10k-labels-idx1-ubyte";

    std::string output_csv = "results/metrics.csv";

    // Local SGD: number of local gradient steps between weight-averaging syncs.
    // 1 = sync every step (equivalent to flat DP), >1 = local SGD.
    int sync_every = 1;
};

}  // namespace nn

#endif  // NN_TRAINING_CONFIG_H_
