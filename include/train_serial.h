#ifndef NN_TRAINING_TRAIN_SERIAL_H_
#define NN_TRAINING_TRAIN_SERIAL_H_

#include <string>

#include "config.h"

namespace nn {

int run_serial_training(const TrainConfig& config, std::string* error_message);

}  // namespace nn

#endif  // NN_TRAINING_TRAIN_SERIAL_H_
