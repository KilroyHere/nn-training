#ifndef NN_TRAINING_TRAIN_COMMON_H_
#define NN_TRAINING_TRAIN_COMMON_H_

#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "data_mnist.h"
#include "mlp.h"

namespace nn {

struct PreparedDatasets {
    Dataset train;
    Dataset val;
    std::vector<int> train_epoch_indices;
};

struct EpochMetrics {
    float train_loss = 0.0f;
    float train_acc = 0.0f;
    float val_loss = 0.0f;
    float val_acc = 0.0f;
    long long epoch_time_ms = 0;
};

void validate_train_config(const TrainConfig& config);
PreparedDatasets prepare_datasets(const TrainConfig& config, std::mt19937* rng);
std::vector<int> build_layer_sizes(const TrainConfig& config);
std::string hidden_layers_csv(const std::vector<int>& hidden_layers);

bool ensure_parent_dir(const std::string& file_path);

}  // namespace nn

#endif  // NN_TRAINING_TRAIN_COMMON_H_
