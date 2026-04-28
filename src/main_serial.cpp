// Serial executable entrypoint.
#include <iostream>
#include <string>

#include "config.h"
#include "train_cli.h"
#include "train_serial.h"

int main(int argc, char** argv) {
    nn::TrainConfig config;
    std::string error_message;
    const nn::CliParseResult parse_result =
        nn::parse_train_args(argc, argv, &config, &error_message);
    if (parse_result == nn::CliParseResult::kHelp) {
        nn::print_train_usage();
        return 0;
    }
    if (parse_result == nn::CliParseResult::kError) {
        std::cerr << error_message << "\n";
        nn::print_train_usage();
        return 1;
    }

    const int rc = nn::run_serial_training(config, &error_message);
    if (rc != 0) {
        std::cerr << "Training failed: " << error_message << "\n";
        return rc;
    }

    std::cout << "Training finished (serial). Metrics written to " << config.output_csv << "\n";
    return 0;
}
