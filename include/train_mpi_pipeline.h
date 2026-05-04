#pragma once

#include <mpi.h>
#include <random>
#include <string>

#include "config.h"

namespace nn {
namespace mpi_mp {

int run_mpi_model_parallel_pipeline(
    const TrainConfig& config,
    int rank,
    int world_size,
    MPI_Comm comm,
    std::string* error_message);

}  // namespace mpi_mp
}  // namespace nn
