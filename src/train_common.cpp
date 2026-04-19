// Shared training setup, dataset prep, and filesystem helper routines.
#include "train_common.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

namespace nn {

namespace {

// Creates a single directory path component if needed.
bool ensure_dir(const std::string& path) {
    if (path.empty()) {
        return true;
    }
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return (info.st_mode & S_IFDIR) != 0;
    }
    return mkdir(path.c_str(), 0755) == 0;
}

}  // namespace

// Validates common training hyperparameters and sample counts.
void validate_train_config(const TrainConfig& config) {
    if (config.batch_size <= 0 || config.epochs <= 0 || config.learning_rate <= 0.0f) {
        throw std::invalid_argument("batch_size, epochs, and learning_rate must be positive");
    }
    if (config.train_samples <= 0 || config.val_samples <= 0) {
        throw std::invalid_argument("train_samples and val_samples must be positive");
    }
}

// Loads MNIST and creates deterministic train/val subsets.
PreparedDatasets prepare_datasets(const TrainConfig& config, std::mt19937* rng) {
    if (rng == nullptr) {
        throw std::invalid_argument("prepare_datasets requires non-null rng");
    }

    Dataset train_full = load_mnist_dataset(config.mnist_train_images, config.mnist_train_labels);
    Dataset test_full = load_mnist_dataset(config.mnist_test_images, config.mnist_test_labels);

    if (config.train_samples > train_full.features.rows) {
        throw std::invalid_argument("train_samples exceeds available MNIST train images");
    }
    if (config.val_samples > test_full.features.rows) {
        throw std::invalid_argument("val_samples exceeds available MNIST test images");
    }

    std::vector<int> train_pick(static_cast<size_t>(train_full.features.rows));
    std::iota(train_pick.begin(), train_pick.end(), 0);
    std::shuffle(train_pick.begin(), train_pick.end(), *rng);
    train_pick.resize(static_cast<size_t>(config.train_samples));

    std::vector<int> val_pick(static_cast<size_t>(test_full.features.rows));
    std::iota(val_pick.begin(), val_pick.end(), 0);
    std::shuffle(val_pick.begin(), val_pick.end(), *rng);
    val_pick.resize(static_cast<size_t>(config.val_samples));

    PreparedDatasets out;
    out.train = subset_dataset(train_full, train_pick);
    out.val = subset_dataset(test_full, val_pick);
    out.train_epoch_indices.resize(static_cast<size_t>(out.train.features.rows));
    std::iota(out.train_epoch_indices.begin(), out.train_epoch_indices.end(), 0);
    return out;
}

// Builds full layer-size vector from config fields.
std::vector<int> build_layer_sizes(const TrainConfig& config) {
    std::vector<int> layer_sizes;
    layer_sizes.push_back(config.input_dim);
    layer_sizes.insert(layer_sizes.end(), config.hidden_layers.begin(), config.hidden_layers.end());
    layer_sizes.push_back(config.num_classes);
    return layer_sizes;
}

// Converts hidden layer sizes to a stable CSV-friendly token.
std::string hidden_layers_csv(const std::vector<int>& hidden_layers) {
    std::string out;
    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        if (i > 0) {
            out += "-";
        }
        out += std::to_string(hidden_layers[i]);
    }
    return out;
}

// Recursively creates parent directory chain for a target file path.
bool ensure_parent_dir(const std::string& file_path) {
    const size_t slash = file_path.find_last_of('/');
    if (slash == std::string::npos) {
        return true;
    }
    const std::string parent = file_path.substr(0, slash);
    if (parent.empty()) {
        return true;
    }

    std::string current = (parent.front() == '/') ? "/" : "";
    size_t start = (parent.front() == '/') ? 1 : 0;

    size_t pos = start;
    while (pos <= parent.size()) {
        size_t next = parent.find('/', pos);
        if (next == std::string::npos) {
            next = parent.size();
        }
        const std::string component = parent.substr(pos, next - pos);
        if (!component.empty()) {
            if (!current.empty() && current.back() != '/') {
                current.push_back('/');
            }
            current += component;
            if (!ensure_dir(current)) {
                return false;
            }
        }
        pos = next + 1;
    }
    return true;
}

}  // namespace nn
