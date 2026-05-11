#include "train_mpi_model_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
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

struct StepTimings {
    double fwd_comm_s    = 0.0;
    double fwd_compute_s = 0.0;
    double fwd_send_s    = 0.0;
    double bwd_comm_s    = 0.0;
    double bwd_compute_s = 0.0;
    double grad_apply_s  = 0.0;
    double bwd_send_s    = 0.0;

    static constexpr int N_FIELDS = 7;

    void pack(double* buf) const {
        buf[0] = fwd_comm_s;
        buf[1] = fwd_compute_s;
        buf[2] = fwd_send_s;
        buf[3] = bwd_comm_s;
        buf[4] = bwd_compute_s;
        buf[5] = grad_apply_s;
        buf[6] = bwd_send_s;
    }

    void accumulate(const StepTimings& o) {
        fwd_comm_s    += o.fwd_comm_s;
        fwd_compute_s += o.fwd_compute_s;
        fwd_send_s    += o.fwd_send_s;
        bwd_comm_s    += o.bwd_comm_s;
        bwd_compute_s += o.bwd_compute_s;
        grad_apply_s  += o.grad_apply_s;
        bwd_send_s    += o.bwd_send_s;
    }
};

void print_epoch_timings(
    const StepTimings& local,
    int rank,
    int world_size,
    int epoch,
    MPI_Comm comm) {

    double local_buf[StepTimings::N_FIELDS];
    local.pack(local_buf);

    std::vector<double> all_buf;
    if (rank == 0) {
        all_buf.resize(static_cast<size_t>(world_size * StepTimings::N_FIELDS), 0.0);
    }

    MPI_Gather(
        local_buf, StepTimings::N_FIELDS, MPI_DOUBLE,
        rank == 0 ? all_buf.data() : nullptr, StepTimings::N_FIELDS, MPI_DOUBLE,
        0, comm);

    if (rank != 0) return;

    static const char* labels[StepTimings::N_FIELDS] = {
        "fwd_comm   ",
        "fwd_compute",
        "fwd_send   ",
        "bwd_comm   ",
        "bwd_compute",
        "grad_apply ",
        "bwd_send   ",
    };

    std::cout << "[mpi-mp] epoch " << epoch
              << " timings (s, epoch total) - min / mean / max across "
              << world_size << " ranks:\n";

    std::cout << std::fixed << std::setprecision(4);
    for (int f = 0; f < StepTimings::N_FIELDS; ++f) {
        double mn  =  std::numeric_limits<double>::max();
        double mx  = -std::numeric_limits<double>::max();
        double sum = 0.0;
        for (int r = 0; r < world_size; ++r) {
            const double v = all_buf[static_cast<size_t>(r * StepTimings::N_FIELDS + f)];
            mn  = std::min(mn,  v);
            mx  = std::max(mx,  v);
            sum += v;
        }
        std::cout << "  " << labels[f] << "  "
                  << std::setw(8) << mn  << " / "
                  << std::setw(8) << (sum / static_cast<double>(world_size)) << " / "
                  << std::setw(8) << mx  << "\n";
    }
    std::cout << std::flush;
}

struct LayerRange {
    int start;
    int end;
};

LayerRange compute_layer_range(int num_layers, int rank, int world_size) {
    if (num_layers < world_size) {
        throw std::invalid_argument(
            "num_layers (" + std::to_string(num_layers) + ") < world_size (" +
            std::to_string(world_size) +
            "): cannot assign at least one layer per rank");
    }
    const int base      = num_layers / world_size;
    const int remainder = num_layers % world_size;
    LayerRange r;
    r.start = rank * base + std::min(rank, remainder);
    r.end   = r.start + base + (rank < remainder ? 1 : 0);
    return r;
}

struct LocalMetrics {
    float loss     = 0.0f;
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
        int   pred = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < probs.cols; ++j) {
            if (probs.at(i, j) > best) {
                best = probs.at(i, j);
                pred = j;
            }
        }
        if (pred == label) ++correct;
    }
    m.loss     /= static_cast<float>(probs.rows);
    m.accuracy  = static_cast<float>(correct) / static_cast<float>(probs.rows);
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
            const int in_dim  = layer_sizes[static_cast<size_t>(i)];
            const int out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            random_normal(in_dim, out_dim,
                          std::sqrt(2.0f / static_cast<float>(in_dim)), rng);
        }
        for (int i = range.start; i < range.end; ++i) {
            const int   in_dim  = layer_sizes[static_cast<size_t>(i)];
            const int   out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            const float stddev  = std::sqrt(2.0f / static_cast<float>(in_dim));
            Layer layer;
            layer.weights = random_normal(in_dim, out_dim, stddev, rng);
            layer.bias    = std::vector<float>(static_cast<size_t>(out_dim), 0.0f);
            layers_.push_back(std::move(layer));
        }
        for (int i = range.end; i < total_layers; ++i) {
            const int in_dim  = layer_sizes[static_cast<size_t>(i)];
            const int out_dim = layer_sizes[static_cast<size_t>(i + 1)];
            random_normal(in_dim, out_dim,
                          std::sqrt(2.0f / static_cast<float>(in_dim)), rng);
        }
    }

    int  input_dim()  const { return input_dim_; }
    int  output_dim() const { return output_dim_; }
    bool is_last()    const { return is_last_; }
    bool is_first()   const { return range_.start == 0; }

    Matrix forward(const Matrix& input) {
        saved_activations_.clear();
        saved_pre_activations_.clear();
        saved_activations_.push_back(input);

        Matrix a = input;
        for (size_t i = 0; i < layers_.size(); ++i) {
            Matrix z = matmul(a, layers_[i].weights);
            add_row_vector(&z, layers_[i].bias);
            saved_pre_activations_.push_back(z);
            if (!(is_last_ && i + 1 == layers_.size())) {
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
            probs.at(i, y[static_cast<size_t>(i)]) -= 1.0f;
        }
        for (float& v : probs.data) v *= inv_batch;

        return backward_from_zgrad(probs, grad_to_prev);
    }

    GradientBuffers backward_nonlast(const Matrix& recv_grad, Matrix* grad_to_prev) {
        if (is_last_) {
            throw std::logic_error("backward_nonlast called on last rank");
        }
        const Matrix grad_z_last =
            relu_backward(recv_grad, saved_pre_activations_.back());
        return backward_from_zgrad(grad_z_last, grad_to_prev);
    }

    void apply_gradients(const GradientBuffers& grads, float lr) {
        if (grads.weight_grads.size() != layers_.size() ||
            grads.bias_grads.size()   != layers_.size()) {
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
    GradientBuffers backward_from_zgrad(
        const Matrix& initial_grad,
        Matrix* grad_to_prev) {
        GradientBuffers grads;
        const size_t n = layers_.size();
        grads.weight_grads.resize(n);
        grads.bias_grads.resize(n);

        Matrix grad = initial_grad;
        for (int li = static_cast<int>(n) - 1; li >= 0; --li) {
            const Matrix& activation = saved_activations_[static_cast<size_t>(li)];
            grads.weight_grads[static_cast<size_t>(li)] =
                matmul(transpose(activation), grad);

            std::vector<float>& bias_grad = grads.bias_grads[static_cast<size_t>(li)];
            bias_grad.assign(static_cast<size_t>(grad.cols), 0.0f);
            for (int i = 0; i < grad.rows; ++i) {
                for (int j = 0; j < grad.cols; ++j) {
                    bias_grad[static_cast<size_t>(j)] += grad.at(i, j);
                }
            }

            if (li == 0) {
                if (grad_to_prev != nullptr) {
                    *grad_to_prev = matmul(
                        grad,
                        transpose(layers_[static_cast<size_t>(li)].weights));
                }
                break;
            }

            const Matrix grad_prev = matmul(
                grad,
                transpose(layers_[static_cast<size_t>(li)].weights));
            grad = relu_backward(
                grad_prev,
                saved_pre_activations_[static_cast<size_t>(li - 1)]);
        }
        return grads;
    }

    LayerRange           range_;
    bool                 is_last_;
    int                  input_dim_;
    int                  output_dim_;
    std::vector<Layer>   layers_;
    std::vector<Matrix>  saved_activations_;
    std::vector<Matrix>  saved_pre_activations_;
};

constexpr int TAG_FWD = 0;
constexpr int TAG_BWD = 1;

void send_activation_blocking(const Matrix& a, int dest, MPI_Comm comm) {
    const int dims[2] = {a.rows, a.cols};
    MPI_Send(dims, 2, MPI_INT, dest, TAG_FWD, comm);
    MPI_Send(a.data.data(), static_cast<int>(a.data.size()),
             MPI_FLOAT, dest, TAG_FWD, comm);
}

Matrix recv_activation_blocking(int src, MPI_Comm comm) {
    int dims[2] = {0, 0};
    MPI_Recv(dims, 2, MPI_INT, src, TAG_FWD, comm, MPI_STATUS_IGNORE);
    Matrix a(dims[0], dims[1], 0.0f);
    MPI_Recv(a.data.data(), static_cast<int>(a.data.size()),
             MPI_FLOAT, src, TAG_FWD, comm, MPI_STATUS_IGNORE);
    return a;
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

// void gather_batch(
//     const Dataset& ds,
//     const std::vector<int>& epoch_indices,
//     int pos,
//     int batch_size,
//     Matrix* x_out,
//     std::vector<int>* y_out) {
//     if (x_out == nullptr || y_out == nullptr) {
//         throw std::invalid_argument("gather_batch requires non-null outputs");
//     }
//     if (x_out->rows != batch_size || x_out->cols != ds.features.cols) {
//         throw std::invalid_argument("gather_batch output matrix shape mismatch");
//     }
//     y_out->resize(static_cast<size_t>(batch_size));
//     for (int i = 0; i < batch_size; ++i) {
//         const int src = epoch_indices[static_cast<size_t>(pos + i)];
//         const size_t dst_row = static_cast<size_t>(i) * static_cast<size_t>(x_out->cols);
//         const size_t src_row = static_cast<size_t>(src) * static_cast<size_t>(ds.features.cols);
//         for (int j = 0; j < ds.features.cols; ++j) {
//             x_out->data[dst_row + static_cast<size_t>(j)] =
//                 ds.features.data[src_row + static_cast<size_t>(j)];
//         }
//         (*y_out)[static_cast<size_t>(i)] = ds.labels[static_cast<size_t>(src)];
//     }
// }

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
    MPI_Comm comm,
    StepTimings* timings) {

    const int batch_rows = x_batch.rows;
    double t0, t1;

    Matrix bwd_recv_buf(batch_rows, slice->output_dim(), 0.0f);
    MPI_Request bwd_recv_req = MPI_REQUEST_NULL;
    if (!slice->is_last()) {
        bwd_recv_req = irecv_data(
            bwd_recv_buf.data.data(),
            static_cast<int>(bwd_recv_buf.data.size()),
            rank + 1, TAG_BWD, comm);
    }

    Matrix local_input;
    if (!slice->is_first()) {
        t0 = MPI_Wtime();
        local_input = recv_activation_blocking(rank - 1, comm);
        t1 = MPI_Wtime();
        if (timings) timings->fwd_comm_s += t1 - t0;
    } else {
        local_input = x_batch;
    }

    t0 = MPI_Wtime();
    Matrix local_output = slice->forward(local_input);
    t1 = MPI_Wtime();
    if (timings) timings->fwd_compute_s += t1 - t0;

    MPI_Request fwd_send_req = MPI_REQUEST_NULL;
    if (!slice->is_last()) {
        const int dims[2] = {local_output.rows, local_output.cols};
        MPI_Send(dims, 2, MPI_INT, rank + 1, TAG_FWD, comm);
        fwd_send_req = isend_data(
            local_output.data.data(),
            static_cast<int>(local_output.data.size()),
            rank + 1, TAG_FWD, comm);
    }

    GradientBuffers grads;
    Matrix          grad_to_prev;
    LocalMetrics    step_metrics;

    if (slice->is_last()) {
        t0 = MPI_Wtime();
        grads = slice->backward_last(
            y_batch,
            slice->is_first() ? nullptr : &grad_to_prev,
            &step_metrics);
        t1 = MPI_Wtime();
        if (timings) timings->bwd_compute_s += t1 - t0;
    } else {
        t0 = MPI_Wtime();
        MPI_Wait(&bwd_recv_req, MPI_STATUS_IGNORE);
        t1 = MPI_Wtime();
        if (timings) timings->bwd_comm_s += t1 - t0;

        t0 = MPI_Wtime();
        grads = slice->backward_nonlast(
            bwd_recv_buf,
            slice->is_first() ? nullptr : &grad_to_prev);
        t1 = MPI_Wtime();
        if (timings) timings->bwd_compute_s += t1 - t0;
    }

    if (fwd_send_req != MPI_REQUEST_NULL) {
        t0 = MPI_Wtime();
        MPI_Wait(&fwd_send_req, MPI_STATUS_IGNORE);
        t1 = MPI_Wtime();
        if (timings) timings->fwd_send_s += t1 - t0;
    }

    t0 = MPI_Wtime();
    slice->apply_gradients(grads, learning_rate);
    t1 = MPI_Wtime();
    if (timings) timings->grad_apply_s += t1 - t0;

    MPI_Request bwd_send_req = MPI_REQUEST_NULL;
    if (!slice->is_first() && !grad_to_prev.data.empty()) {
        bwd_send_req = isend_data(
            grad_to_prev.data.data(),
            static_cast<int>(grad_to_prev.data.size()),
            rank - 1, TAG_BWD, comm);
    }
    if (bwd_send_req != MPI_REQUEST_NULL) {
        t0 = MPI_Wtime();
        MPI_Wait(&bwd_send_req, MPI_STATUS_IGNORE);
        t1 = MPI_Wtime();
        if (timings) timings->bwd_send_s += t1 - t0;
    }

    StepResult result{step_metrics.loss, step_metrics.accuracy};
    float metrics_buf[2] = {result.train_loss, result.train_acc};
    MPI_Bcast(metrics_buf, 2, MPI_FLOAT, world_size - 1, comm);
    result.train_loss = metrics_buf[0];
    result.train_acc  = metrics_buf[1];
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
    Matrix x_batch(batch_size, val.features.cols, 0.0f);
    std::vector<int> y_batch(static_cast<size_t>(batch_size));

    for (int pos = 0; pos + batch_size <= val.features.rows; pos += batch_size) {
        for (int i = 0; i < batch_size; ++i) {
            const size_t dst_row = static_cast<size_t>(i) * static_cast<size_t>(x_batch.cols);
            const size_t src_row =
                static_cast<size_t>(pos + i) * static_cast<size_t>(val.features.cols);
            for (int j = 0; j < val.features.cols; ++j) {
                x_batch.data[dst_row + static_cast<size_t>(j)] =
                    val.features.data[src_row + static_cast<size_t>(j)];
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
            const LocalMetrics lm = compute_local_metrics(probs, y_batch);
            sum_loss += lm.loss;
            sum_acc  += lm.accuracy;
            ++steps;
        }
    }

    ValResult vr{0.0f, 0.0f};
    if (slice->is_last() && steps > 0) {
        vr.val_loss = sum_loss / static_cast<float>(steps);
        vr.val_acc  = sum_acc  / static_cast<float>(steps);
    }

    float vbuf[2] = {vr.val_loss, vr.val_acc};
    MPI_Bcast(vbuf, 2, MPI_FLOAT, world_size - 1, comm);
    vr.val_loss = vbuf[0];
    vr.val_acc  = vbuf[1];
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
    int epoch,
    MPI_Comm comm) {
    if (slice == nullptr || epoch_indices == nullptr || rng == nullptr) {
        throw std::invalid_argument("run_mp_epoch requires non-null pointers");
    }

    std::shuffle(epoch_indices->begin(), epoch_indices->end(), *rng);

    MPI_Barrier(comm);
    const auto wall_t0 = std::chrono::high_resolution_clock::now();

    float sum_loss = 0.0f;
    float sum_acc  = 0.0f;
    int   steps    = 0;

    StepTimings epoch_timings;
    Matrix x_batch(config.batch_size, train.features.cols, 0.0f);
    std::vector<int> y_batch(static_cast<size_t>(config.batch_size));

    for (int pos = 0; pos + config.batch_size <= train.features.rows; pos += config.batch_size) {
        gather_batch(train, *epoch_indices, pos, config.batch_size, &x_batch, &y_batch);

        StepTimings step_timings;
        const StepResult step_result = run_mp_step(
            slice, x_batch, y_batch, config.learning_rate,
            rank, world_size, comm, &step_timings);

        epoch_timings.accumulate(step_timings);
        sum_loss += step_result.train_loss;
        sum_acc  += step_result.train_acc;
        ++steps;
    }

    if (steps == 0) {
        throw std::runtime_error(
            "No training steps executed; check batch_size and dataset size");
    }

    // Stop clock here — epoch_time_ms is training only, val eval excluded.
    MPI_Barrier(comm);
    const auto wall_t1 = std::chrono::high_resolution_clock::now();

    print_epoch_timings(epoch_timings, rank, world_size, epoch, comm);

    const ValResult vr = run_mp_eval(slice, val, config.batch_size, rank, world_size, comm);

    EpochMetrics out;
    out.train_loss    = sum_loss / static_cast<float>(steps);
    out.train_acc     = sum_acc  / static_cast<float>(steps);
    out.val_loss      = vr.val_loss;
    out.val_acc       = vr.val_acc;
    out.epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            wall_t1 - wall_t0).count();
    return out;
}

}  // anonymous namespace

int run_mpi_model_parallel_training(
    const TrainConfig& config,
    std::string* error_message) {
    try {
        validate_train_config(config);

        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error(
                "MPI must be initialized before calling model-parallel training");
        }

        int rank       = 0;
        int world_size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        const std::vector<int> layer_sizes = build_layer_sizes(config);
        const int num_layers = static_cast<int>(layer_sizes.size()) - 1;
        if (num_layers < world_size) {
            throw std::invalid_argument(
                "Model parallelism requires num_layers (" +
                std::to_string(num_layers) + ") >= world_size (" +
                std::to_string(world_size) + ")");
        }

        const LayerRange range        = compute_layer_range(num_layers, rank, world_size);
        const bool       is_last_rank = (rank == world_size - 1);

        std::mt19937     rng(config.seed);
        PreparedDatasets datasets = prepare_datasets(config, &rng);

        MPLayerSlice slice(layer_sizes, range, is_last_rank, rng);

        for (int r = 0; r < world_size; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " owns layers ["
                          << range.start << ", " << range.end << ")\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        std::ofstream out;
        const std::string hidden_str = hidden_layers_csv(config.hidden_layers);
        if (rank == 0) {
            if (!ensure_parent_dir(config.output_csv)) {
                throw std::runtime_error(
                    "Failed to create output directory for: " + config.output_csv);
            }
            out.open(config.output_csv);
            if (!out.is_open()) {
                throw std::runtime_error(
                    "Failed to open output file: " + config.output_csv);
            }
            out << "mode,seed,learning_rate,batch_size,train_samples,val_samples,"
                   "hidden_layers,epoch,train_loss,train_acc,val_loss,val_acc,"
                   "epoch_time_ms\n";
        }

        for (int epoch = 1; epoch <= config.epochs; ++epoch) {
            const EpochMetrics epoch_metrics = run_mp_epoch(
                &slice, config, datasets.train, datasets.val,
                &datasets.train_epoch_indices, &rng,
                rank, world_size, epoch, MPI_COMM_WORLD);

            if (rank == 0) {
                out << "mpi-mp,"        << config.seed          << ","
                    << config.learning_rate                      << ","
                    << config.batch_size                         << ","
                    << config.train_samples                      << ","
                    << config.val_samples                        << ","
                    << hidden_str                                << ","
                    << epoch                                     << ","
                    << epoch_metrics.train_loss                  << ","
                    << epoch_metrics.train_acc                   << ","
                    << epoch_metrics.val_loss                    << ","
                    << epoch_metrics.val_acc                     << ","
                    << epoch_metrics.epoch_time_ms               << "\n";

                std::cout << "[mpi-mp] epoch " << epoch << "/" << config.epochs
                          << " time_ms="    << epoch_metrics.epoch_time_ms
                          << " train_loss=" << epoch_metrics.train_loss
                          << " train_acc="  << epoch_metrics.train_acc
                          << " val_loss="   << epoch_metrics.val_loss
                          << " val_acc="    << epoch_metrics.val_acc
                          << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& ex) {
        if (error_message != nullptr) *error_message = ex.what();
        return 1;
    }
}

}  // namespace nn