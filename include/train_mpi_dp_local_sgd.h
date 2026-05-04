#ifndef NN_TRAINING_TRAIN_MPI_DP_LOCAL_SGD_H_
#define NN_TRAINING_TRAIN_MPI_DP_LOCAL_SGD_H_

#include <string>

#include "config.h"

namespace nn {

// Runs Local SGD data-parallel training.
// Each rank performs config.sync_every local SGD steps before averaging model
// weights across all ranks via MPI_Allreduce. Gradient communication is
// eliminated between syncs, reducing communication cost by ~sync_every×.
// config.sync_every == 1 is equivalent to flat data-parallel SGD.
int run_mpi_dp_local_sgd_training(const TrainConfig& config, std::string* error_message);

}  // namespace nn

#endif  // NN_TRAINING_TRAIN_MPI_DP_LOCAL_SGD_H_
