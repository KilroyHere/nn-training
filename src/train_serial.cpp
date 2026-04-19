#include "train_serial.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "train_common.h"

namespace nn {

namespace {

Matrix gather_rows(const Matrix& m, const std::vector<int>& indices, int start, int count) {
    Matrix out(count, m.cols, 0.0f);
    for (int i = 0; i < count; ++i) {
        const int src = indices[static_cast<size_t>(start + i)];
        for (int j = 0; j < m.cols; ++j) {
            out.at(i, j) = m.at(src, j);
        }
    }
    return out;
}

std::vector<int> gather_labels(
    const std::vector<int>& labels,
    const std::vector<int>& indices,
    int start,
    int count) {
    std::vector<int> out(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] =
            labels[static_cast<size_t>(indices[static_cast<size_t>(start + i)])];
    }
    return out;
}

EpochMetrics run_serial_epoch(
    MLP* model,
    const TrainConfig& config,
    const Dataset& train,
    const Dataset& val,
    std::vector<int>* epoch_indices,
    std::mt19937* rng) {
    if (model == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_serial_epoch requires non-null pointers");
    }

    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);
    const auto start = std::chrono::high_resolution_clock::now();

    float running_loss = 0.0f;
    float running_acc = 0.0f;
    int steps = 0;
    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        const Matrix x_batch = gather_rows(train.features, *epoch_indices, pos, config.batch_size);
        const std::vector<int> y_batch =
            gather_labels(train.labels, *epoch_indices, pos, config.batch_size);
        const BatchMetrics train_metrics = model->train_batch(x_batch, y_batch, config.learning_rate);
        running_loss += train_metrics.loss;
        running_acc += train_metrics.accuracy;
        ++steps;
    }

    if (steps == 0) {
        throw std::runtime_error("No train steps executed; batch_size too large?");
    }

    const BatchMetrics val_metrics = model->evaluate_batch(val.features, val.labels);
    const auto end = std::chrono::high_resolution_clock::now();

    EpochMetrics out;
    out.train_loss = running_loss / static_cast<float>(steps);
    out.train_acc = running_acc / static_cast<float>(steps);
    out.val_loss = val_metrics.loss;
    out.val_acc = val_metrics.accuracy;
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return out;
}

}  // namespace

int run_serial_training(const TrainConfig& config, std::string* error_message) {
    try {
        validate_train_config(config);

        std::mt19937 rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);
        const std::vector<int> layer_sizes = build_layer_sizes(config);
        MLP mlp(layer_sizes, rng);

        if (!ensure_parent_dir(config.output_csv)) {
            throw std::runtime_error("Failed to create output directory for: " + config.output_csv);
        }

        std::ofstream out(config.output_csv);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open output file: " + config.output_csv);
        }
        out << "mode,seed,learning_rate,batch_size,train_samples,val_samples,hidden_layers,"
               "epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms\n";
        const std::string hidden_layers = hidden_layers_csv(config.hidden_layers);

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            const EpochMetrics epoch_metrics = run_serial_epoch(
                &mlp,
                config,
                datasets.train,
                datasets.val,
                &datasets.train_epoch_indices,
                &rng);
            out << "serial,"
                << config.seed << ","
                << config.learning_rate << ","
                << config.batch_size << ","
                << config.train_samples << ","
                << config.val_samples << ","
                << hidden_layers << ","
                << epoch << ","
                << epoch_metrics.train_loss << ","
                << epoch_metrics.train_acc << ","
                << epoch_metrics.val_loss << ","
                << epoch_metrics.val_acc << ","
                << epoch_metrics.epoch_time_ms << "\n";

            std::cout << "[serial] epoch " << epoch << "/" << config.epochs
                      << " time_ms=" << epoch_metrics.epoch_time_ms
                      << " train_loss=" << epoch_metrics.train_loss
                      << " train_acc=" << epoch_metrics.train_acc
                      << " val_loss=" << epoch_metrics.val_loss
                      << " val_acc=" << epoch_metrics.val_acc << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        if (error_message != nullptr) {
            *error_message = ex.what();
        }
        return 1;
    }
}

}  // namespace nn
