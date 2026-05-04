// Local SGD data-parallel training: ranks update independently for sync_every
// steps then average model weights via a single fused MPI_Allreduce.
#include "train_mpi_dp_local_sgd.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "mlp.h"
#include "train_common.h"

namespace nn {

namespace {

// Copies selected feature rows into a contiguous batch matrix.
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

// Copies selected labels into a contiguous batch vector.
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

// Returns a contiguous submatrix of rows [start, start+count).
Matrix slice_rows(const Matrix& m, int start, int count) {
    Matrix out(count, m.cols, 0.0f);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            out.at(i, j) = m.at(start + i, j);
        }
    }
    return out;
}

// Returns a contiguous slice of labels [start, start+count).
std::vector<int> slice_labels(const std::vector<int>& labels, int start, int count) {
    return std::vector<int>(labels.begin() + start, labels.begin() + start + count);
}

// Averages model weights across all ranks using one fused MPI_Allreduce.
// Returns time spent in MPI_Allreduce in seconds (excludes pack/unpack).
double average_weights(MLP* model, std::vector<float>* weight_buf, int world_size, MPI_Comm comm) {
    const size_t n = weight_buf->size();
    model->pack_weights(weight_buf->data());

    const double t0 = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, weight_buf->data(), static_cast<int>(n), MPI_FLOAT, MPI_SUM, comm);
    const double elapsed = MPI_Wtime() - t0;

    const float inv_world = 1.0f / static_cast<float>(world_size);
    for (float& v : *weight_buf) {
        v *= inv_world;
    }
    model->unpack_weights(weight_buf->data());
    return elapsed;
}

// Runs one Local SGD epoch.
// Each rank updates locally for sync_every steps, then all ranks average weights.
// A final sync is forced at epoch end if the last step was not a sync boundary.
EpochMetrics run_mpi_dp_local_sgd_epoch(
    MLP* model,
    const TrainConfig& config,
    const Dataset& train,
    const Dataset& val,
    std::vector<int>* epoch_indices,
    std::mt19937* rng,
    std::vector<float>* weight_buf,
    int rank,
    int world_size,
    MPI_Comm comm) {
    if (model == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_mpi_dp_local_sgd_epoch requires non-null pointers");
    }
    if (config.batch_size % world_size != 0) {
        throw std::invalid_argument("batch_size must be divisible by MPI world size");
    }
    if (config.sync_every < 1) {
        throw std::invalid_argument("sync_every must be >= 1");
    }

    const int local_batch = config.batch_size / world_size;
    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);

    MPI_Barrier(comm);
    const auto start = std::chrono::high_resolution_clock::now();

    float local_running_loss = 0.0f;
    float local_running_acc = 0.0f;
    int local_steps = 0;
    double total_compute_s = 0.0;
    double total_sync_s = 0.0;
    int sync_count = 0;

    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        const int local_start = pos + (rank * local_batch);
        const Matrix x_batch =
            gather_rows(train.features, *epoch_indices, local_start, local_batch);
        const std::vector<int> y_batch =
            gather_labels(train.labels, *epoch_indices, local_start, local_batch);

        // Local SGD step — no communication.
        const double t_compute = MPI_Wtime();
        const BatchMetrics local_metrics =
            model->train_batch(x_batch, y_batch, config.learning_rate);
        total_compute_s += MPI_Wtime() - t_compute;

        local_running_loss += local_metrics.loss;
        local_running_acc += local_metrics.accuracy;
        ++local_steps;

        // Weight average every sync_every steps.
        if (local_steps % config.sync_every == 0) {
            total_sync_s += average_weights(model, weight_buf, world_size, comm);
            ++sync_count;
        }
    }

    if (local_steps == 0) {
        throw std::runtime_error("No train steps executed; batch_size too large?");
    }

    // Force final sync if last step was not a sync boundary.
    if (local_steps % config.sync_every != 0) {
        total_sync_s += average_weights(model, weight_buf, world_size, comm);
        ++sync_count;
    }

    // Aggregate per-rank training loss/acc into global means.
    double loss_acc_steps[3] = {
        static_cast<double>(local_running_loss),
        static_cast<double>(local_running_acc),
        static_cast<double>(local_steps)};
    double global_loss_acc_steps[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(loss_acc_steps, global_loss_acc_steps, 3, MPI_DOUBLE, MPI_SUM, comm);

    // Stop clock — epoch_time_ms is training only, val eval excluded.
    MPI_Barrier(comm);
    const auto end = std::chrono::high_resolution_clock::now();

    EpochMetrics out;
    out.train_loss = static_cast<float>(global_loss_acc_steps[0] / global_loss_acc_steps[2]);
    out.train_acc = static_cast<float>(global_loss_acc_steps[1] / global_loss_acc_steps[2]);
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Distributed val eval (outside training timer).
    const int val_per_rank = val.features.rows / world_size;
    const int val_start = rank * val_per_rank;
    double val_loss_sum = 0.0;
    double val_correct_sum = 0.0;
    if (val_per_rank > 0) {
        const Matrix x_val_local = slice_rows(val.features, val_start, val_per_rank);
        const std::vector<int> y_val_local = slice_labels(val.labels, val_start, val_per_rank);
        const BatchMetrics local_val = model->evaluate_batch(x_val_local, y_val_local);
        val_loss_sum = static_cast<double>(local_val.loss) * val_per_rank;
        val_correct_sum = static_cast<double>(local_val.accuracy) * val_per_rank;
    }
    double val_stats[3] = {val_loss_sum, val_correct_sum, static_cast<double>(val_per_rank)};
    double global_val_stats[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(val_stats, global_val_stats, 3, MPI_DOUBLE, MPI_SUM, comm);
    out.val_loss = static_cast<float>(global_val_stats[0] / global_val_stats[2]);
    out.val_acc = static_cast<float>(global_val_stats[1] / global_val_stats[2]);

    if (rank == 0) {
        const double compute_ms = total_compute_s * 1e3;
        const double sync_ms = total_sync_s * 1e3;
        const double other_ms = out.epoch_time_ms - compute_ms - sync_ms;
        std::cout << "[mpi-dp-local-sgd][timing] "
                  << "compute_ms=" << compute_ms
                  << " sync_ms=" << sync_ms
                  << " syncs_per_epoch=" << sync_count
                  << " other_ms=" << other_ms
                  << std::endl;
    }
    return out;
}

}  // namespace

// Top-level Local SGD training runner and CSV writer.
int run_mpi_dp_local_sgd_training(const TrainConfig& config, std::string* error_message) {
    try {
        validate_train_config(config);
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI must be initialized before calling local SGD training");
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size <= 0) {
            throw std::runtime_error("Invalid MPI world size");
        }
        if (config.batch_size % world_size != 0) {
            throw std::invalid_argument("batch_size must be divisible by MPI world size");
        }
        if (config.sync_every < 1) {
            throw std::invalid_argument("sync_every must be >= 1");
        }

        std::mt19937 rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);
        const std::vector<int> layer_sizes = build_layer_sizes(config);
        MLP mlp(layer_sizes, rng);

        // Pre-allocate weight buffer once; reused every sync to avoid per-call allocation.
        std::vector<float> weight_buf(mlp.weight_buffer_size());

        std::ofstream out;
        const std::string hidden_layers = hidden_layers_csv(config.hidden_layers);
        if (rank == 0) {
            if (!ensure_parent_dir(config.output_csv)) {
                throw std::runtime_error("Failed to create output directory for: " + config.output_csv);
            }
            out.open(config.output_csv);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to open output file: " + config.output_csv);
            }
            out << "mode,seed,learning_rate,batch_size,train_samples,val_samples,hidden_layers,"
                   "epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms\n";
            std::cout << "[mpi-dp-local-sgd] world_size=" << world_size
                      << " sync_every=" << config.sync_every
                      << " weight_params=" << weight_buf.size() << std::endl;
        }

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            const EpochMetrics epoch_metrics = run_mpi_dp_local_sgd_epoch(
                &mlp,
                config,
                datasets.train,
                datasets.val,
                &datasets.train_epoch_indices,
                &rng,
                &weight_buf,
                rank,
                world_size,
                MPI_COMM_WORLD);

            if (rank == 0) {
                out << "mpi-dp-local-sgd,"
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

                std::cout << "[mpi-dp-local-sgd] epoch " << epoch << "/" << config.epochs
                          << " time_ms=" << epoch_metrics.epoch_time_ms
                          << " train_loss=" << epoch_metrics.train_loss
                          << " train_acc=" << epoch_metrics.train_acc
                          << " val_loss=" << epoch_metrics.val_loss
                          << " val_acc=" << epoch_metrics.val_acc << std::endl;
            }
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
