cmake_minimum_required(VERSION 3.24)
project(WeatherSimCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Default to Blackwell RTX 5060 (sm_120); can override via -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(WeatherSimCuda SHARED
    src/weather_sim_api.cpp
    src/weather_kernels.cu
)

target_include_directories(WeatherSimCuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(WeatherSimCuda PRIVATE
    CUDA::cudart_static
)

target_compile_definitions(WeatherSimCuda PRIVATE
    WEATHER_SIM_CUDA_EXPORTS
)

target_compile_options(WeatherSimCuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /O2 /W3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,-O2 --use_fast_math>
)

set_target_properties(WeatherSimCuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
