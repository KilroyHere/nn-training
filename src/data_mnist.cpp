#include "data_mnist.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace nn {

namespace {

uint32_t read_be_u32(std::ifstream& in) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) {
        throw std::runtime_error("Failed reading MNIST header");
    }
    return (static_cast<uint32_t>(bytes[0]) << 24U) |
           (static_cast<uint32_t>(bytes[1]) << 16U) |
           (static_cast<uint32_t>(bytes[2]) << 8U) |
           static_cast<uint32_t>(bytes[3]);
}

}  // namespace

Dataset load_mnist_dataset(const std::string& image_path, const std::string& label_path, int max_samples) {
    std::ifstream image_in(image_path, std::ios::binary);
    std::ifstream label_in(label_path, std::ios::binary);
    if (!image_in.is_open() || !label_in.is_open()) {
        throw std::runtime_error(
            "Failed opening MNIST files: " + image_path + " and " + label_path);
    }

    const uint32_t image_magic = read_be_u32(image_in);
    const uint32_t image_count = read_be_u32(image_in);
    const uint32_t rows = read_be_u32(image_in);
    const uint32_t cols = read_be_u32(image_in);
    const uint32_t label_magic = read_be_u32(label_in);
    const uint32_t label_count = read_be_u32(label_in);

    if (image_magic != 2051U) {
        throw std::runtime_error("Invalid MNIST image magic for file: " + image_path);
    }
    if (label_magic != 2049U) {
        throw std::runtime_error("Invalid MNIST label magic for file: " + label_path);
    }
    if (image_count != label_count) {
        throw std::runtime_error("MNIST images/labels count mismatch");
    }

    const uint32_t available = std::min(image_count, label_count);
    const uint32_t to_read =
        max_samples > 0 ? std::min<uint32_t>(available, static_cast<uint32_t>(max_samples)) : available;
    const int input_dim = static_cast<int>(rows * cols);

    Dataset dataset;
    dataset.features = Matrix(static_cast<int>(to_read), input_dim, 0.0f);
    dataset.labels.resize(static_cast<size_t>(to_read), 0);

    std::vector<unsigned char> image_row(static_cast<size_t>(input_dim), 0);
    for (uint32_t i = 0; i < to_read; ++i) {
        image_in.read(reinterpret_cast<char*>(image_row.data()), image_row.size());
        if (!image_in) {
            throw std::runtime_error("Unexpected EOF in MNIST images");
        }
        for (int j = 0; j < input_dim; ++j) {
            dataset.features.at(static_cast<int>(i), j) =
                static_cast<float>(image_row[static_cast<size_t>(j)]) / 255.0f;
        }
        unsigned char label = 0;
        label_in.read(reinterpret_cast<char*>(&label), 1);
        if (!label_in) {
            throw std::runtime_error("Unexpected EOF in MNIST labels");
        }
        dataset.labels[static_cast<size_t>(i)] = static_cast<int>(label);
    }

    return dataset;
}

Dataset subset_dataset(const Dataset& full, const std::vector<int>& indices) {
    if (full.features.rows != static_cast<int>(full.labels.size())) {
        throw std::runtime_error("Invalid dataset: rows != labels");
    }
    Dataset out;
    out.features = Matrix(static_cast<int>(indices.size()), full.features.cols, 0.0f);
    out.labels.resize(indices.size(), 0);

    const int input_dim = full.features.cols;
    for (size_t i = 0; i < indices.size(); ++i) {
        const int src = indices[i];
        if (src < 0 || src >= full.features.rows) {
            throw std::out_of_range("subset index out of range");
        }
        for (int j = 0; j < input_dim; ++j) {
            out.features.at(static_cast<int>(i), j) = full.features.at(src, j);
        }
        out.labels[i] = full.labels[static_cast<size_t>(src)];
    }
    return out;
}

}  // namespace nn
