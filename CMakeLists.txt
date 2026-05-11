cmake_minimum_required(VERSION 3.16)
project(nn_training LANGUAGES CXX)

option(NN_FORCE_MODERN_GCC "Require GCC >= 9 for reliable C++17 support" ON)
option(NN_ENABLE_NATIVE_TUNING "Enable -march=native in Release builds" ON)
option(NN_ENABLE_IPO "Enable interprocedural optimization in Release builds" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NN_FORCE_MODERN_GCC AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(FATAL_ERROR
            "Detected GCC ${CMAKE_CXX_COMPILER_VERSION}, but nn_training requires GCC >= 9 when NN_FORCE_MODERN_GCC=ON. "
            "Reconfigure with a newer compiler, e.g. CXX=/usr/bin/g++ cmake -S . -B build")
    endif()
endif()

message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<CONFIG:Release>:-O3>)
    if(NN_ENABLE_NATIVE_TUNING)
        add_compile_options($<$<CONFIG:Release>:-march=native>)
    endif()
endif()

if(NN_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT nn_ipo_supported OUTPUT nn_ipo_error)
    if(nn_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(STATUS "IPO disabled: ${nn_ipo_error}")
    endif()
endif()

add_library(nn_core
    src/tensor.cpp
    src/data_mnist.cpp
    src/mlp.cpp
    src/train_cli.cpp
    src/train_common.cpp
    src/train_serial.cpp
)

target_include_directories(nn_core PUBLIC include)
target_link_libraries(nn_core PUBLIC ${BLAS_LIBRARIES})

add_executable(serial_train src/main_serial.cpp)
target_link_libraries(serial_train PRIVATE nn_core)

add_executable(mpi_dp_train src/main_mpi_dp.cpp src/train_mpi_data_parallel.cpp)
target_link_libraries(mpi_dp_train PRIVATE nn_core MPI::MPI_CXX)

add_executable(
    mpi_dp_hierarchial_train
    src/main_mpi_dp_hierarchial.cpp
    src/train_mpi_dp_hierarchial.cpp)
target_link_libraries(mpi_dp_hierarchial_train PRIVATE nn_core MPI::MPI_CXX)

add_executable(mpi_mp_train src/main_mpi_mp.cpp src/train_mpi_model_parallel.cpp)
target_link_libraries(mpi_mp_train PRIVATE nn_core MPI::MPI_CXX)

add_executable(
    mpi_dp_local_sgd_train
    src/main_mpi_dp_local_sgd.cpp
    src/train_mpi_dp_local_sgd.cpp)
target_link_libraries(mpi_dp_local_sgd_train PRIVATE nn_core MPI::MPI_CXX)
