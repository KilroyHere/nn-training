#ifndef NN_TRAINING_TRAIN_MPI_DP_HIERARCHIAL_H_
#define NN_TRAINING_TRAIN_MPI_DP_HIERARCHIAL_H_

#include <string>

#include "config.h"

namespace nn {

int run_mpi_dp_hierarchial_training(const TrainConfig& config, std::string* error_message);

}  // namespace nn

#endif  // NN_TRAINING_TRAIN_MPI_DP_HIERARCHIAL_H_
