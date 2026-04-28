#ifndef NN_TRAINING_TRAIN_CLI_H_
#define NN_TRAINING_TRAIN_CLI_H_

#include <string>

#include "config.h"

namespace nn {

enum class CliParseResult { kOk, kHelp, kError };

void print_train_usage();
CliParseResult parse_train_args(
    int argc,
    char** argv,
    TrainConfig* config,
    std::string* error_message);

}  // namespace nn

#endif  // NN_TRAINING_TRAIN_CLI_H_
