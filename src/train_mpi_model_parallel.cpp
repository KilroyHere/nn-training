#include "train_mpi_model_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "mlp.h"
#include "tensor.h"
#include "train_common.h"

namespace nn {

namespace {

struct LayerRange {
    int start;
    int end;
};

LayerRange compute_layer_range(int num_layers, int rank, int world_size) {
    if (num_layers < world_size) {
        throw std::invalid_argument(
            "num_layers (" + std::to_string(num_layers) + ") < world_size (" +
            std::to_string(world_size) + "): cannot assign at least one layer per rank");
    }
    const int base = num_layers / world_size;
    const int remainder = num_layers % world_size;
    LayerRange r;
    r.start = rank * base + std::min(rank, remainder);
    r.end = r.start + base + (rank < remainder ? 1 : 0);
    return r;
}

struct LocalMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
};

LocalMetrics compute_local_metrics(const Matrix& probs, const std::vector<int>& y) {
    if (probs.rows != static_cast<int>(y.size())) {
        throw std::invalid_argument("compute_local_metrics: dimension mismatch");
    }
    LocalMetrics m;
    int correct = 0;
    for (int i = 0; i < probs.rows; ++i) {
        const int label = y[static_cast<size_t>(i)];
        m.loss += -std::log(std::max(probs.at(i, label), 1e-8f));

        int pred = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < probs.cols; ++j) {
            if (probs.at(i, j) > best) {
                best = probs.at(i, j);
                pred = j;
            }
        }
        if (pred == label) {
            ++correct;
        }
    }
    m.loss /= static_cast<float>(probs.rows);
    m.accuracy = static_cast<float>(correct) / static_cast<float>(probs.rows);
    return m;
}

class MPLayerSlice {
public:
    MPLayerSlice(
        const std::vector<int>& layer_sizes,
        const LayerRange& range,
        bool is_last_rank,
        std::mt19937& rng)
        : range_(range),
          is_last_(is_last_rank),
          input_dim_(layer_sizes[static_cast<size_t>(range.start)]),
          output_dim_(layer_sizes[static_cast<size_t>(range.end)]) {

        if (range.start >= range.end) {
            throw std::invalid_argument("MPLayerSlice requires a non-empty layer range");
        }
        const int total_layers = static_cast<int>(layer_sizes.size()) - 1;

        for (int i = 0; i < range.start; ++i) {
            const int in_dim = layer_sizes[static_cast<size_t>(i)];
            const int out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            random_normal(in_dim, out_dim, std::sqrt(2.0f / static_cast<float>(in_dim)), rng);
        }

        for (int i = range.start; i < range.end; ++i) {
            const int in_dim = layer_sizes[static_cast<size_t>(i)];
            const int out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            const float stddev = std::sqrt(2.0f / static_cast<float>(in_dim));
            Layer layer;
            layer.weights = random_normal(in_dim, out_dim, stddev, rng);
            layer.bias = std::vector<float>(static_cast<size_t>(out_dim), 0.0f);
            layers_.push_back(std::move(layer));
        }

        for (int i = range.end; i < total_layers; ++i) {
            const int in_dim = layer_sizes[static_cast<size_t>(i)];
            const int out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            random_normal(in_dim, out_dim, std::sqrt(2.0f / static_cast<float>(in_dim)), rng);
        }
    }

    int input_dim() const { return input_dim_; }
    int output_dim() const { return output_dim_; }
    bool is_last() const { return is_last_; }
    bool is_first() const { return range_.start == 0; }

    Matrix forward(const Matrix& input) {
        saved_activations_.clear();
        saved_pre_activations_.clear();
        saved_activations_.push_back(input);

        Matrix a = input;
        for (size_t i = 0; i < layers_.size(); ++i) {
            Matrix z = matmul(a, layers_[i].weights);
            add_row_vector(&z, layers_[i].bias);
            saved_pre_activations_.push_back(z);
            if (is_last_ && i + 1 == layers_.size()) {
                // pass
            } else {
                relu_inplace(&z);
            }
            a = std::move(z);
            saved_activations_.push_back(a);
        }
        return saved_activations_.back();
    }

    GradientBuffers backward_last(
        const std::vector<int>& y,
        Matrix* grad_to_prev,
        LocalMetrics* metrics_out) {
        if (!is_last_) {
            throw std::logic_error("backward_last called on non-last rank");
        }
        const Matrix& logits = saved_activations_.back();
        Matrix probs = softmax_rows(logits);

        if (metrics_out != nullptr) {
            *metrics_out = compute_local_metrics(probs, y);
        }

        const float inv_batch = 1.0f / static_cast<float>(probs.rows);
        for (int i = 0; i < probs.rows; ++i) {
            const int label = y[static_cast<size_t>(i)];
            probs.at(i, label) -= 1.0f;
        }
        for (float& v : probs.data) {
            v *= inv_batch;
        }

        return backward_from_zgrad(probs, grad_to_prev);
    }

    GradientBuffers backward_nonlast(const Matrix& recv_grad, Matrix* grad_to_prev) {
        if (is_last_) {
            throw std::logic_error("backward_nonlast called on last rank");
        }
        const Matrix grad_z_last = relu_backward(recv_grad, saved_pre_activations_.back());
        return backward_from_zgrad(grad_z_last, grad_to_prev);
    }

    void apply_gradients(const GradientBuffers& grads, float lr) {
        if (grads.weight_grads.size() != layers_.size() ||
            grads.bias_grads.size() != layers_.size()) {
            throw std::invalid_argument("Gradient sizes do not match local layer count");
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            Matrix& weights = layers_[i].weights;
            const Matrix& grad_w = grads.weight_grads[i];
            if (weights.rows != grad_w.rows || weights.cols != grad_w.cols) {
                throw std::invalid_argument("Weight gradient shape mismatch");
            }
            for (size_t j = 0; j < weights.data.size(); ++j) {
                weights.data[j] -= lr * grad_w.data[j];
            }
            std::vector<float>& bias = layers_[i].bias;
            const std::vector<float>& bias_grad = grads.bias_grads[i];
            if (bias.size() != bias_grad.size()) {
                throw std::invalid_argument("Bias gradient shape mismatch");
            }
            for (size_t j = 0; j < bias.size(); ++j) {
                bias[j] -= lr * bias_grad[j];
            }
        }
    }

private:
    GradientBuffers backward_from_zgrad(const Matrix& initial_grad, Matrix* grad_to_prev) {
        GradientBuffers grads;
        const size_t n = layers_.size();
        grads.weight_grads.resize(n);
        grads.bias_grads.resize(n);

        Matrix grad = initial_grad;
        for (int li = static_cast<int>(n) - 1; li >= 0; --li) {
            const Matrix& activation = saved_activations_[static_cast<size_t>(li)];
            grads.weight_grads[static_cast<size_t>(li)] = matmul(transpose(activation), grad);

            std::vector<float>& bias_grad = grads.bias_grads[static_cast<size_t>(li)];
            bias_grad.assign(static_cast<size_t>(grad.cols), 0.0f);
            for (int i = 0; i < grad.rows; ++i) {
                for (int j = 0; j < grad.cols; ++j) {
                    bias_grad[static_cast<size_t>(j)] += grad.at(i, j);
                }
            }

            if (li == 0) {
                if (grad_to_prev != nullptr) {
                    const Matrix grad_prev = matmul(grad, transpose(layers_[static_cast<size_t>(li)].weights));
                    *grad_to_prev = std::move(grad_prev);
                }
                break;
            }

            const Matrix grad_prev = matmul(grad, transpose(layers_[static_cast<size_t>(li)].weights));
            grad = relu_backward(grad_prev, saved_pre_activations_[static_cast<size_t>(li - 1)]);
        }
        return grads;
    }

    LayerRange range_;
    bool is_last_;
    int input_dim_;
    int output_dim_;
    std::vector<Layer> layers_;
    std::vector<Matrix> saved_activations_;
    std::vector<Matrix> saved_pre_activations_;
};

constexpr int TAG_FWD = 0;
constexpr int TAG_BWD = 1;

void send_activation_blocking(const Matrix& a, int dest, MPI_Comm comm) {
    const int dims[2] = {a.rows, a.cols};
    MPI_Send(dims, 2, MPI_INT, dest, TAG_FWD, comm);
    MPI_Send(a.data.data(), static_cast<int>(a.data.size()), MPI_FLOAT, dest, TAG_FWD, comm);
}

Matrix recv_activation_blocking(int src, MPI_Comm comm) {
    int dims[2] = {0, 0};
    MPI_Recv(dims, 2, MPI_INT, src, TAG_FWD, comm, MPI_STATUS_IGNORE);
    Matrix a(dims[0], dims[1], 0.0f);
    MPI_Recv(a.data.data(), static_cast<int>(a.data.size()), MPI_FLOAT, src, TAG_FWD, comm, MPI_STATUS_IGNORE);
    return a;
}

void send_gradient_blocking(const Matrix& g, int dest, MPI_Comm comm) {
    MPI_Send(g.data.data(), static_cast<int>(g.data.size()), MPI_FLOAT, dest, TAG_BWD, comm);
}

Matrix recv_gradient_blocking(int rows, int cols, int src, MPI_Comm comm) {
    Matrix g(rows, cols, 0.0f);
    MPI_Recv(g.data.data(), static_cast<int>(g.data.size()), MPI_FLOAT, src, TAG_BWD, comm, MPI_STATUS_IGNORE);
    return g;
}

MPI_Request isend_data(const float* buf, int count, int dest, int tag, MPI_Comm comm) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(buf, count, MPI_FLOAT, dest, tag, comm, &req);
    return req;
}

MPI_Request irecv_data(float* buf, int count, int src, int tag, MPI_Comm comm) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Irecv(buf, count, MPI_FLOAT, src, tag, comm, &req);
    return req;
}

void gather_batch(
    const Dataset& ds,
    const std::vector<int>& epoch_indices,
    int pos,
    int batch_size,
    Matrix* x_out,
    std::vector<int>* y_out) {
    *x_out = Matrix(batch_size, ds.features.cols, 0.0f);
    y_out->resize(static_cast<size_t>(batch_size));
    for (int i = 0; i < batch_size; ++i) {
        const int src = epoch_indices[static_cast<size_t>(pos + i)];
        for (int j = 0; j < ds.features.cols; ++j) {
            x_out->at(i, j) = ds.features.at(src, j);
        }
        (*y_out)[static_cast<size_t>(i)] = ds.labels[static_cast<size_t>(src)];
    }
}

struct StepResult {
    float train_loss;
    float train_acc;
};

StepResult run_mp_step(
    MPLayerSlice* slice,
    const Matrix& x_batch,
    const std::vector<int>& y_batch,
    float learning_rate,
    int rank,
    int world_size,
    MPI_Comm comm) {
    const int batch_rows = x_batch.rows;

    Matrix bwd_recv_buf(batch_rows, slice->output_dim(), 0.0f);
    MPI_Request bwd_recv_req = MPI_REQUEST_NULL;
    if (!slice->is_last()) {
        bwd_recv_req = irecv_data(
            bwd_recv_buf.data.data(),
            static_cast<int>(bwd_recv_buf.data.size()),
            rank + 1,
            TAG_BWD,
            comm);
    }

    Matrix local_input;
    MPI_Request fwd_send_req = MPI_REQUEST_NULL;

    if (!slice->is_first()) {
        local_input = recv_activation_blocking(rank - 1, comm);
    } else {
        local_input = x_batch;
    }

    Matrix local_output = slice->forward(local_input);
    if (!slice->is_last()) {
        const int dims[2] = {local_output.rows, local_output.cols};
        MPI_Send(dims, 2, MPI_INT, rank + 1, TAG_FWD, comm);
        fwd_send_req = isend_data(
            local_output.data.data(),
            static_cast<int>(local_output.data.size()),
            rank + 1,
            TAG_FWD,
            comm);
    }

    GradientBuffers grads;
    Matrix grad_to_prev;
    LocalMetrics step_metrics;

    if (slice->is_last()) {
        grads = slice->backward_last(y_batch, slice->is_first() ? nullptr : &grad_to_prev, &step_metrics);
    } else {
        MPI_Wait(&bwd_recv_req, MPI_STATUS_IGNORE);
        grads = slice->backward_nonlast(bwd_recv_buf, slice->is_first() ? nullptr : &grad_to_prev);
    }

    if (fwd_send_req != MPI_REQUEST_NULL) {
        MPI_Wait(&fwd_send_req, MPI_STATUS_IGNORE);
    }

    MPI_Request bwd_send_req = MPI_REQUEST_NULL;
    if (!slice->is_first() && !grad_to_prev.data.empty()) {
        bwd_send_req = isend_data(
            grad_to_prev.data.data(),
            static_cast<int>(grad_to_prev.data.size()),
            rank - 1,
            TAG_BWD,
            comm);
    }

    slice->apply_gradients(grads, learning_rate);

    if (bwd_send_req != MPI_REQUEST_NULL) {
        MPI_Wait(&bwd_send_req, MPI_STATUS_IGNORE);
    }

    StepResult result{step_metrics.loss, step_metrics.accuracy};
    float metrics_buf[2] = {result.train_loss, result.train_acc};
    MPI_Bcast(metrics_buf, 2, MPI_FLOAT, world_size - 1, comm);
    result.train_loss = metrics_buf[0];
    result.train_acc = metrics_buf[1];
    return result;
}

struct ValResult {
    float val_loss;
    float val_acc;
};

ValResult run_mp_eval(
    MPLayerSlice* slice,
    const Dataset& val,
    int batch_size,
    int rank,
    int world_size,
    MPI_Comm comm) {
    float sum_loss = 0.0f;
    float sum_acc = 0.0f;
    int steps = 0;

    for (int pos = 0; pos + batch_size <= val.features.rows; pos += batch_size) {
        Matrix x_batch(batch_size, val.features.cols, 0.0f);
        std::vector<int> y_batch(static_cast<size_t>(batch_size));
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < val.features.cols; ++j) {
                x_batch.at(i, j) = val.features.at(pos + i, j);
            }
            y_batch[static_cast<size_t>(i)] = val.labels[static_cast<size_t>(pos + i)];
        }

        Matrix local_input;
        if (!slice->is_first()) {
            local_input = recv_activation_blocking(rank - 1, comm);
        } else {
            local_input = x_batch;
        }

        Matrix local_output = slice->forward(local_input);
        if (!slice->is_last()) {
            send_activation_blocking(local_output, rank + 1, comm);
        }

        if (slice->is_last()) {
            const Matrix probs = softmax_rows(local_output);
            const LocalMetrics local_metrics = compute_local_metrics(probs, y_batch);
            sum_loss += local_metrics.loss;
            sum_acc += local_metrics.accuracy;
            ++steps;
        }
    }

    ValResult vr{0.0f, 0.0f};
    if (slice->is_last() && steps > 0) {
        vr.val_loss = sum_loss / static_cast<float>(steps);
        vr.val_acc = sum_acc / static_cast<float>(steps);
    }

    float vbuf[2] = {vr.val_loss, vr.val_acc};
    MPI_Bcast(vbuf, 2, MPI_FLOAT, world_size - 1, comm);
    vr.val_loss = vbuf[0];
    vr.val_acc = vbuf[1];
    return vr;
}

EpochMetrics run_mp_epoch(
    MPLayerSlice* slice,
    const TrainConfig& config,
    const Dataset& train,
    const Dataset& val,
    std::vector<int>* epoch_indices,
    std::mt19937* rng,
    int rank,
    int world_size,
    MPI_Comm comm) {
    if (slice == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_mp_epoch requires non-null pointers");
    }

    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);

    MPI_Barrier(comm);
    const auto t0 = std::chrono::high_resolution_clock::now();

    float sum_loss = 0.0f;
    float sum_acc = 0.0f;
    int steps = 0;

    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        Matrix x_batch;
        std::vector<int> y_batch;
        gather_batch(train, *epoch_indices, pos, config.batch_size, &x_batch, &y_batch);

        const StepResult step_result = run_mp_step(
            slice,
            x_batch,
            y_batch,
            config.learning_rate,
            rank,
            world_size,
            comm);

        sum_loss += step_result.train_loss;
        sum_acc += step_result.train_acc;
        ++steps;
    }

    if (steps == 0) {
        throw std::runtime_error("No training steps executed; check batch_size and dataset size");
    }

    const ValResult vr = run_mp_eval(slice, val, config.batch_size, rank, world_size, comm);

    MPI_Barrier(comm);
    const auto t1 = std::chrono::high_resolution_clock::now();

    EpochMetrics out;
    out.train_loss = sum_loss / static_cast<float>(steps);
    out.train_acc = sum_acc / static_cast<float>(steps);
    out.val_loss = vr.val_loss;
    out.val_acc = vr.val_acc;
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return out;
}

}

int run_mpi_model_parallel_training(const TrainConfig& config, std::string* error_message) {
    try {
        validate_train_config(config);

        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("MPI must be initialized before calling model-parallel training");
        }

        int rank = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        const std::vector<int> layer_sizes = build_layer_sizes(config);
        const int num_layers = static_cast<int>(layer_sizes.size()) - 1;
        if (num_layers < world_size) {
            throw std::invalid_argument(
                "Model parallelism requires num_layers (" + std::to_string(num_layers) + ") >= world_size (" +
                std::to_string(world_size) + ")");
        }

        const LayerRange range = compute_layer_range(num_layers, rank, world_size);
        const bool is_last_rank = (rank == world_size - 1);

        std::mt19937 rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);

        MPLayerSlice slice(layer_sizes, range, is_last_rank, rng);

        for (int r = 0; r < world_size; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " owns layers [" << range.start << ", " << range.end << ")\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::ofstream out;
        const std::string hidden_str = hidden_layers_csv(config.hidden_layers);
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
            const EpochMetrics epoch_metrics = run_mp_epoch(
                &slice,
                config,
                datasets.train,
                datasets.val,
                &datasets.train_epoch_indices,
                &rng,
                rank,
                world_size,
                MPI_COMM_WORLD);

            if (rank == 0) {
                out << "mpi-mp," << config.seed << "," << config.learning_rate << ","
                    << config.batch_size << "," << config.train_samples << ","
                    << config.val_samples << "," << hidden_str << "," << epoch << ","
                    << epoch_metrics.train_loss << "," << epoch_metrics.train_acc << ","
                    << epoch_metrics.val_loss << "," << epoch_metrics.val_acc << ","
                    << epoch_metrics.epoch_time_ms << "\n";

                std::cout << "[mpi-mp] epoch " << epoch << "/" << config.epochs
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

}
