#pragma once

#include <string>
#include "config.h"

namespace nn {
    int run_mpi_model_parallel_training(const TrainConfig& config, std::string* error_message);
}
