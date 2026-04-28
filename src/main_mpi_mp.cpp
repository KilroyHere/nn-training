// MPI model-parallel executable entrypoint.
#include <iostream>
#include <mpi.h>
#include <string>

#include "config.h"
#include "train_cli.h"
#include "train_mpi_model_parallel.h"

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

    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int rc = nn::run_mpi_model_parallel_training(config, &error_message);
    if (rc != 0 && rank == 0) {
        std::cerr << "Training failed: " << error_message << "\n";
    }

    MPI_Finalize();
    if (rc != 0) {
        return rc;
    }

    if (rank == 0) {
        std::cout << "Training finished (mpi-mp). Metrics written to " << config.output_csv << "\n";
    }
    return 0;
}
