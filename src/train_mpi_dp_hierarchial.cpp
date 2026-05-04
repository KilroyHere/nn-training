// MPI data-parallel hierarchical training loop and gradient synchronization logic.
#include "train_mpi_dp_hierarchial.h"

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

struct HierarchicalComms {
    MPI_Comm intra_node_comm = MPI_COMM_NULL;
    MPI_Comm inter_node_comm = MPI_COMM_NULL;
    int local_rank = 0;
    int local_size = 1;
    int world_rank = 0;
    int world_size = 1;
    int inter_size = 0;
    int node_count = 1;
    int min_local_size = 1;
    int max_local_size = 1;
    bool is_leader = true;
};

// Timing breakdown for one epoch's gradient synchronization.
struct HierarchicalTimings {
    double intra_reduce_s = 0.0;
    double inter_allreduce_s = 0.0;
    double intra_bcast_s = 0.0;
    double compute_s = 0.0;
};

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

// Initializes communicators for hierarchical collectives.
HierarchicalComms create_hierarchical_comms(MPI_Comm world_comm) {
    HierarchicalComms comms;
    MPI_Comm_rank(world_comm, &comms.world_rank);
    MPI_Comm_size(world_comm, &comms.world_size);
    if (comms.world_size <= 0) {
        throw std::runtime_error("Invalid MPI world size");
    }

    MPI_Comm_split_type(
        world_comm,
        MPI_COMM_TYPE_SHARED,
        comms.world_rank,
        MPI_INFO_NULL,
        &comms.intra_node_comm);
    if (comms.intra_node_comm == MPI_COMM_NULL) {
        throw std::runtime_error("Failed to create intra-node communicator");
    }

    MPI_Comm_rank(comms.intra_node_comm, &comms.local_rank);
    MPI_Comm_size(comms.intra_node_comm, &comms.local_size);
    comms.is_leader = (comms.local_rank == 0);

    const int leader_color = comms.is_leader ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(world_comm, leader_color, comms.world_rank, &comms.inter_node_comm);
    if (comms.is_leader && comms.inter_node_comm == MPI_COMM_NULL) {
        throw std::runtime_error("Failed to create inter-node leader communicator");
    }
    if (comms.is_leader) {
        MPI_Comm_size(comms.inter_node_comm, &comms.inter_size);
    }

    const int local_is_leader = comms.is_leader ? 1 : 0;
    MPI_Allreduce(&local_is_leader, &comms.node_count, 1, MPI_INT, MPI_SUM, world_comm);

    MPI_Allreduce(&comms.local_size, &comms.min_local_size, 1, MPI_INT, MPI_MIN, world_comm);
    MPI_Allreduce(&comms.local_size, &comms.max_local_size, 1, MPI_INT, MPI_MAX, world_comm);
    return comms;
}

// Releases hierarchical communicators.
void destroy_hierarchical_comms(HierarchicalComms* comms) {
    if (comms == nullptr) {
        return;
    }
    if (comms->inter_node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comms->inter_node_comm);
    }
    if (comms->intra_node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comms->intra_node_comm);
    }
}

// Hierarchically reduces gradients across all ranks using fused buffers, then averages.
// All weight and bias gradients are packed into one flat buffer so the three collective
// phases (intra Reduce → inter Allreduce → intra Bcast) each issue a single MPI call
// regardless of layer count, minimising per-call latency and enabling larger-message
// algorithms in the MPI runtime.
void allreduce_gradients_hierarchial(
    GradientBuffers* gradients,
    const HierarchicalComms& comms,
    HierarchicalTimings* timings) {
    if (gradients == nullptr) {
        throw std::invalid_argument("allreduce_gradients_hierarchial requires non-null gradients");
    }

    // Compute total gradient element count and pack into one contiguous buffer.
    size_t total = 0;
    for (size_t i = 0; i < gradients->weight_grads.size(); ++i) {
        total += gradients->weight_grads[i].data.size();
        total += gradients->bias_grads[i].size();
    }
    std::vector<float> buf(total);
    size_t offset = 0;
    for (size_t i = 0; i < gradients->weight_grads.size(); ++i) {
        std::copy(gradients->weight_grads[i].data.begin(),
                  gradients->weight_grads[i].data.end(),
                  buf.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += gradients->weight_grads[i].data.size();
        std::copy(gradients->bias_grads[i].begin(),
                  gradients->bias_grads[i].end(),
                  buf.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += gradients->bias_grads[i].size();
    }
    const int count = static_cast<int>(total);

    // Phase 1: one intra-node Reduce to local root (rank 0 within node).
    double t0 = MPI_Wtime();
    MPI_Reduce(
        comms.is_leader ? MPI_IN_PLACE : buf.data(),
        buf.data(),
        count,
        MPI_FLOAT,
        MPI_SUM,
        0,
        comms.intra_node_comm);
    if (timings != nullptr) {
        timings->intra_reduce_s += MPI_Wtime() - t0;
    }

    // Phase 2: one inter-node Allreduce among node leaders only.
    // Skipped when there is only one node (inter_size == 1).
    if (comms.is_leader && comms.inter_size > 1) {
        double t1 = MPI_Wtime();
        MPI_Allreduce(
            MPI_IN_PLACE,
            buf.data(),
            count,
            MPI_FLOAT,
            MPI_SUM,
            comms.inter_node_comm);
        if (timings != nullptr) {
            timings->inter_allreduce_s += MPI_Wtime() - t1;
        }
    }

    // Phase 3: one intra-node Bcast from local root to all ranks on node.
    double t2 = MPI_Wtime();
    MPI_Bcast(buf.data(), count, MPI_FLOAT, 0, comms.intra_node_comm);
    if (timings != nullptr) {
        timings->intra_bcast_s += MPI_Wtime() - t2;
    }

    // Unpack fused buffer back into per-layer gradient tensors, scaling to global mean.
    const float inv_world = 1.0f / static_cast<float>(comms.world_size);
    offset = 0;
    for (size_t i = 0; i < gradients->weight_grads.size(); ++i) {
        for (float& v : gradients->weight_grads[i].data) {
            v = buf[offset++] * inv_world;
        }
        for (float& v : gradients->bias_grads[i]) {
            v = buf[offset++] * inv_world;
        }
    }
}

// Runs one MPI-DP hierarchial epoch with fixed global-batch semantics.
EpochMetrics run_mpi_dp_hierarchial_epoch(
    MLP* model,
    const TrainConfig& config,
    const Dataset& train,
    const Dataset& val,
    std::vector<int>* epoch_indices,
    std::mt19937* rng,
    const HierarchicalComms& comms) {
    if (model == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_mpi_dp_hierarchial_epoch requires non-null pointers");
    }
    if (config.batch_size % comms.world_size != 0) {
        throw std::invalid_argument("batch_size must be divisible by MPI world size");
    }

    const int local_batch = config.batch_size / comms.world_size;
    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    float local_running_loss = 0.0f;
    float local_running_acc = 0.0f;
    int local_steps = 0;
    HierarchicalTimings timings;
    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        const int local_start = pos + (comms.world_rank * local_batch);
        const Matrix x_batch = gather_rows(train.features, *epoch_indices, local_start, local_batch);
        const std::vector<int> y_batch =
            gather_labels(train.labels, *epoch_indices, local_start, local_batch);

        GradientBuffers gradients;
        double t_compute = MPI_Wtime();
        const BatchMetrics local_metrics = model->compute_batch_gradients(x_batch, y_batch, &gradients);
        timings.compute_s += MPI_Wtime() - t_compute;

        allreduce_gradients_hierarchial(&gradients, comms, &timings);
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
    MPI_Allreduce(loss_acc_steps, global_loss_acc_steps, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Stop clock here — epoch_time_ms covers only the training loop + train metric reduce.
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    EpochMetrics out;
    out.train_loss = static_cast<float>(global_loss_acc_steps[0] / global_loss_acc_steps[2]);
    out.train_acc = static_cast<float>(global_loss_acc_steps[1] / global_loss_acc_steps[2]);
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Distributed val eval (outside training timer).
    const int val_per_rank = val.features.rows / comms.world_size;
    const int val_start = comms.world_rank * val_per_rank;
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
    MPI_Allreduce(val_stats, global_val_stats, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    out.val_loss = static_cast<float>(global_val_stats[0] / global_val_stats[2]);
    out.val_acc = static_cast<float>(global_val_stats[1] / global_val_stats[2]);

    if (comms.world_rank == 0) {
        const double compute_ms = timings.compute_s * 1e3;
        const double intra_reduce_ms = timings.intra_reduce_s * 1e3;
        const double inter_allreduce_ms = timings.inter_allreduce_s * 1e3;
        const double intra_bcast_ms = timings.intra_bcast_s * 1e3;
        const double other_ms =
            out.epoch_time_ms - compute_ms - intra_reduce_ms - inter_allreduce_ms - intra_bcast_ms;
        std::cout << "[mpi-dp-hierarchial][timing] "
                  << "compute_ms=" << compute_ms
                  << " intra_reduce_ms=" << intra_reduce_ms
                  << " inter_allreduce_ms=" << inter_allreduce_ms
                  << " intra_bcast_ms=" << intra_bcast_ms
                  << " other_ms=" << other_ms
                  << std::endl;
    }
    return out;
}

}  // namespace

// Top-level MPI data-parallel hierarchial training runner and CSV writer.
int run_mpi_dp_hierarchial_training(const TrainConfig& config, std::string* error_message) {
    HierarchicalComms comms;
    bool comms_initialized = false;
    try {
        validate_train_config(config);
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI must be initialized before calling DP training");
        }

        comms = create_hierarchical_comms(MPI_COMM_WORLD);
        comms_initialized = true;

        if (config.batch_size % comms.world_size != 0) {
            throw std::invalid_argument("batch_size must be divisible by MPI world size");
        }

        std::mt19937 rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);
        const std::vector<int> layer_sizes = build_layer_sizes(config);
        MLP mlp(layer_sizes, rng);

        std::ofstream out;
        const std::string hidden_layers = hidden_layers_csv(config.hidden_layers);
        if (comms.world_rank == 0) {
            if (!ensure_parent_dir(config.output_csv)) {
                throw std::runtime_error("Failed to create output directory for: " + config.output_csv);
            }
            out.open(config.output_csv);
            if (!out.is_open()) {
                throw std::runtime_error("Failed to open output file: " + config.output_csv);
            }
            out << "mode,seed,learning_rate,batch_size,train_samples,val_samples,hidden_layers,"
                   "epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_ms\n";
            std::cout << "[mpi_dp_hierarchial] world_size=" << comms.world_size
                      << " node_count=" << comms.node_count
                      << " local_size_min=" << comms.min_local_size
                      << " local_size_max=" << comms.max_local_size << std::endl;
        }

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            const EpochMetrics epoch_metrics = run_mpi_dp_hierarchial_epoch(
                &mlp,
                config,
                datasets.train,
                datasets.val,
                &datasets.train_epoch_indices,
                &rng,
                comms);

            if (comms.world_rank == 0) {
                out << "mpi_dp_hierarchial,"
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

                std::cout << "[mpi-dp-hierarchial] epoch " << epoch << "/" << config.epochs
                          << " time_ms=" << epoch_metrics.epoch_time_ms
                          << " train_loss=" << epoch_metrics.train_loss
                          << " train_acc=" << epoch_metrics.train_acc
                          << " val_loss=" << epoch_metrics.val_loss
                          << " val_acc=" << epoch_metrics.val_acc << std::endl;
            }
        }

        destroy_hierarchical_comms(&comms);
        return 0;
    } catch (const std::exception& ex) {
        if (comms_initialized) {
            destroy_hierarchical_comms(&comms);
        }
        if (error_message != nullptr) {
            *error_message = ex.what();
        }
        return 1;
    }
}

}  // namespace nn
