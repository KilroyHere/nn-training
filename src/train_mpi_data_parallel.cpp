// MPI data-parallel training loop and gradient synchronization logic.
#include "train_mpi_data_parallel.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numeric>
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
    return std::vector<int>(
        labels.begin() + start,
        labels.begin() + start + count);
}

// All-reduces gradients across ranks and converts sums to means.
// Returns time spent in MPI_Allreduce calls in seconds.
double allreduce_gradients(GradientBuffers* gradients, int world_size, MPI_Comm comm) {
    if (gradients == nullptr) {
        throw std::invalid_argument("allreduce_gradients requires non-null gradients");
    }
    double allreduce_elapsed = 0.0;
    for (size_t i = 0; i < gradients->weight_grads.size(); ++i) {
        Matrix& grad_w = gradients->weight_grads[i];
        std::vector<float>& grad_b = gradients->bias_grads[i];

        double t0 = MPI_Wtime();
        MPI_Allreduce(
            MPI_IN_PLACE,
            grad_w.data.data(),
            static_cast<int>(grad_w.data.size()),
            MPI_FLOAT,
            MPI_SUM,
            comm);
        MPI_Allreduce(
            MPI_IN_PLACE,
            grad_b.data(),
            static_cast<int>(grad_b.size()),
            MPI_FLOAT,
            MPI_SUM,
            comm);
        allreduce_elapsed += MPI_Wtime() - t0;
    }

    const float inv_world = 1.0f / static_cast<float>(world_size);
    for (size_t i = 0; i < gradients->weight_grads.size(); ++i) {
        Matrix& grad_w = gradients->weight_grads[i];
        std::vector<float>& grad_b = gradients->bias_grads[i];
        for (float& v : grad_w.data) {
            v *= inv_world;
        }
        for (float& v : grad_b) {
            v *= inv_world;
        }
    }
    return allreduce_elapsed;
}

// Runs one MPI-DP epoch with fixed global-batch semantics.
EpochMetrics run_mpi_dp_epoch(
    MLP* model,
    const TrainConfig& config,
    const Dataset& train,
    const Dataset& val,
    std::vector<int>* epoch_indices,
    std::mt19937* rng,
    int rank,
    int world_size,
    MPI_Comm comm) {
    if (model == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_mpi_dp_epoch requires non-null pointers");
    }
    if (config.batch_size % world_size != 0) {
        throw std::invalid_argument("batch_size must be divisible by MPI world size");
    }

    const int local_batch = config.batch_size / world_size;
    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);

    // Barrier-align timing so epoch time reflects global wall-clock progress.
    MPI_Barrier(comm);
    const auto start = std::chrono::high_resolution_clock::now();

    float local_running_loss = 0.0f;
    float local_running_acc = 0.0f;
    int local_steps = 0;
    double total_compute_s = 0.0;
    double total_allreduce_s = 0.0;
    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        const int local_start = pos + (rank * local_batch);
        const Matrix x_batch = gather_rows(train.features, *epoch_indices, local_start, local_batch);
        const std::vector<int> y_batch = gather_labels(train.labels, *epoch_indices, local_start, local_batch);

        GradientBuffers gradients;
        double t_compute = MPI_Wtime();
        const BatchMetrics local_metrics = model->compute_batch_gradients(x_batch, y_batch, &gradients);
        total_compute_s += MPI_Wtime() - t_compute;

        total_allreduce_s += allreduce_gradients(&gradients, world_size, comm);
        model->apply_gradients(gradients, config.learning_rate);

        local_running_loss += local_metrics.loss;
        local_running_acc += local_metrics.accuracy;
        ++local_steps;
    }

    if (local_steps == 0) {
        throw std::runtime_error("No train steps executed; batch_size too large?");
    }

    // Aggregate per-rank training loss/acc into global means.
    double loss_acc_steps[3] = {
        static_cast<double>(local_running_loss),
        static_cast<double>(local_running_acc),
        static_cast<double>(local_steps)};
    double global_loss_acc_steps[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(loss_acc_steps, global_loss_acc_steps, 3, MPI_DOUBLE, MPI_SUM, comm);

    // Stop clock here — epoch_time_ms covers only the training loop + train metric reduce.
    MPI_Barrier(comm);
    const auto end = std::chrono::high_resolution_clock::now();

    EpochMetrics out;
    out.train_loss = static_cast<float>(global_loss_acc_steps[0] / global_loss_acc_steps[2]);
    out.train_acc = static_cast<float>(global_loss_acc_steps[1] / global_loss_acc_steps[2]);
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Distributed val eval (outside training timer).
    // Each rank evaluates its contiguous slice; tail samples are dropped.
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
        const double allreduce_ms = total_allreduce_s * 1e3;
        const double other_ms = out.epoch_time_ms - compute_ms - allreduce_ms;
        std::cout << "[mpi-dp][timing] "
                  << "compute_ms=" << compute_ms
                  << " allreduce_ms=" << allreduce_ms
                  << " other_ms=" << other_ms
                  << std::endl;
    }
    return out;
}

}  // namespace

// Top-level MPI data-parallel training runner and CSV writer.
int run_mpi_data_parallel_training(const TrainConfig& config, std::string* error_message) {
    try {
        validate_train_config(config);
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI must be initialized before calling DP training");
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

        std::mt19937 rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);
        const std::vector<int> layer_sizes = build_layer_sizes(config);
        MLP mlp(layer_sizes, rng);

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
        }

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            const EpochMetrics epoch_metrics = run_mpi_dp_epoch(
                &mlp,
                config,
                datasets.train,
                datasets.val,
                &datasets.train_epoch_indices,
                &rng,
                rank,
                world_size,
                MPI_COMM_WORLD);

            if (rank == 0) {
                out << "mpi,"
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

                std::cout << "[mpi-dp] epoch " << epoch << "/" << config.epochs
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
