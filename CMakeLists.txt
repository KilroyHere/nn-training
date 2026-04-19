cmake_minimum_required(VERSION 3.16)
project(nn_training LANGUAGES CXX)

option(NN_FORCE_MODERN_GCC "Require GCC >= 9 for reliable C++17 support" ON)

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

add_library(nn_core
    src/tensor.cpp
    src/data_mnist.cpp
    src/mlp.cpp
    src/train_common.cpp
    src/train_serial.cpp
)

target_include_directories(nn_core PUBLIC include)
target_link_libraries(nn_core PUBLIC ${BLAS_LIBRARIES})

add_executable(serial_train src/main.cpp)
target_link_libraries(serial_train PRIVATE nn_core)
target_compile_definitions(serial_train PRIVATE NN_ENABLE_MPI=0)

add_executable(mpi_dp_train src/main.cpp src/train_mpi_data_parallel.cpp)
target_link_libraries(mpi_dp_train PRIVATE nn_core MPI::MPI_CXX)
target_compile_definitions(mpi_dp_train PRIVATE NN_ENABLE_MPI=1)
