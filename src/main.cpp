// CLI entrypoint that dispatches to serial or MPI-DP training modes.
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#if NN_ENABLE_MPI
#include <mpi.h>
#include "train_mpi_data_parallel.h"
#endif
#include "train_serial.h"

namespace {

// Prints CLI flags for serial and MPI-DP execution modes.
void print_usage() {
    std::cout << "Usage: train [options]\n"
              << "Options:\n"
              << "  --mode <serial|mpi-dp>\n"
              << "  --epochs <int>\n"
              << "  --batch <int>\n"
              << "  --lr <float>\n"
              << "  --seed <int>\n"
              << "  --train-samples <int>\n"
              << "  --val-samples <int>\n"
              << "  --hidden <comma-separated ints>\n"
              << "  --data-dir <path>\n"
              << "  --output <path>\n";
}

// Parses hidden sizes from a comma-separated string.
bool parse_hidden_layers(const std::string& text, std::vector<int>* out) {
    if (out == nullptr) {
        return false;
    }
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            return false;
        }
        values.push_back(std::stoi(token));
    }
    if (values.empty()) {
        return false;
    }
    *out = values;
    return true;
}

// Defaults mode from binary name to keep wrappers simple.
std::string infer_default_mode(const std::string& argv0) {
    return argv0.find("mpi_dp") != std::string::npos ? "mpi-dp" : "serial";
}

}  // namespace

// Parses CLI args and dispatches into the selected backend.
int main(int argc, char** argv) {
    nn::TrainConfig config;
    std::string mode = infer_default_mode(argc > 0 ? argv[0] : "");
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto consume_value = [&](std::string* value) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            ++i;
            *value = argv[i];
            return true;
        };

        std::string value;
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--mode" && consume_value(&value)) {
            mode = value;
        } else if (arg == "--epochs" && consume_value(&value)) {
            config.epochs = std::stoi(value);
        } else if (arg == "--batch" && consume_value(&value)) {
            config.batch_size = std::stoi(value);
        } else if (arg == "--lr" && consume_value(&value)) {
            config.learning_rate = std::stof(value);
        } else if (arg == "--seed" && consume_value(&value)) {
            config.seed = static_cast<unsigned int>(std::stoul(value));
        } else if (arg == "--train-samples" && consume_value(&value)) {
            config.train_samples = std::stoi(value);
        } else if (arg == "--val-samples" && consume_value(&value)) {
            config.val_samples = std::stoi(value);
        } else if (arg == "--data-dir" && consume_value(&value)) {
            config.mnist_train_images = value + "/train-images-idx3-ubyte";
            config.mnist_train_labels = value + "/train-labels-idx1-ubyte";
            config.mnist_test_images = value + "/t10k-images-idx3-ubyte";
            config.mnist_test_labels = value + "/t10k-labels-idx1-ubyte";
        } else if (arg == "--output" && consume_value(&value)) {
            config.output_csv = value;
        } else if (arg == "--hidden" && consume_value(&value)) {
            if (!parse_hidden_layers(value, &config.hidden_layers)) {
                std::cerr << "Invalid --hidden list: " << value << "\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown or malformed argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (mode != "serial" && mode != "mpi-dp") {
        std::cerr << "Unsupported mode: " << mode << "\n";
        print_usage();
        return 1;
    }

    std::string error_message;
    int rc = 0;
    bool mpi_initialized_here = false;
    int mpi_rank = 0;
    if (mode == "serial") {
        rc = nn::run_serial_training(config, &error_message);
    } else {
#if NN_ENABLE_MPI
        // Main owns MPI lifecycle so backend code can assume initialized MPI.
        MPI_Init(&argc, &argv);
        mpi_initialized_here = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        rc = nn::run_mpi_data_parallel_training(config, &error_message);
#else
        std::cerr << "This binary was built without MPI support.\n";
        return 1;
#endif
    }
    if (rc != 0) {
#if NN_ENABLE_MPI
        if (mode == "mpi-dp" && mpi_initialized_here) {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                std::cerr << "Training failed: " << error_message << "\n";
            }
        } else {
            std::cerr << "Training failed: " << error_message << "\n";
        }
#else
        std::cerr << "Training failed: " << error_message << "\n";
#endif
        if (mpi_initialized_here) {
#if NN_ENABLE_MPI
            MPI_Finalize();
#endif
        }
        return rc;
    }

    if (mpi_initialized_here) {
#if NN_ENABLE_MPI
        MPI_Finalize();
#endif
    }

    if (mode != "mpi-dp" || mpi_rank == 0) {
        std::cout << "Training finished (" << mode << "). Metrics written to " << config.output_csv
                  << "\n";
    }
    return 0;
}
