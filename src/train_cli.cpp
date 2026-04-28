// Shared CLI parsing for train executables.
#include "train_cli.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace nn {

namespace {

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

}  // namespace

void print_train_usage() {
    std::cout << "Usage: train [options]\n"
              << "Options:\n"
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

CliParseResult parse_train_args(
    int argc,
    char** argv,
    TrainConfig* config,
    std::string* error_message) {
    if (config == nullptr || error_message == nullptr) {
        return CliParseResult::kError;
    }
    try {
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
                return CliParseResult::kHelp;
            } else if (arg == "--epochs" && consume_value(&value)) {
                config->epochs = std::stoi(value);
            } else if (arg == "--batch" && consume_value(&value)) {
                config->batch_size = std::stoi(value);
            } else if (arg == "--lr" && consume_value(&value)) {
                config->learning_rate = std::stof(value);
            } else if (arg == "--seed" && consume_value(&value)) {
                config->seed = static_cast<unsigned int>(std::stoul(value));
            } else if (arg == "--train-samples" && consume_value(&value)) {
                config->train_samples = std::stoi(value);
            } else if (arg == "--val-samples" && consume_value(&value)) {
                config->val_samples = std::stoi(value);
            } else if (arg == "--data-dir" && consume_value(&value)) {
                config->mnist_train_images = value + "/train-images-idx3-ubyte";
                config->mnist_train_labels = value + "/train-labels-idx1-ubyte";
                config->mnist_test_images = value + "/t10k-images-idx3-ubyte";
                config->mnist_test_labels = value + "/t10k-labels-idx1-ubyte";
            } else if (arg == "--output" && consume_value(&value)) {
                config->output_csv = value;
            } else if (arg == "--hidden" && consume_value(&value)) {
                if (!parse_hidden_layers(value, &config->hidden_layers)) {
                    *error_message = "Invalid --hidden list: " + value;
                    return CliParseResult::kError;
                }
            } else {
                *error_message = "Unknown or malformed argument: " + arg;
                return CliParseResult::kError;
            }
        }
    } catch (const std::exception& ex) {
        *error_message = std::string("Argument parse error: ") + ex.what();
        return CliParseResult::kError;
    }
    return CliParseResult::kOk;
}

}  // namespace nn
