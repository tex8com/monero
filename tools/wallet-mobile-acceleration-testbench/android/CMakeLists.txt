cmake_minimum_required(VERSION 3.22)
project(wallet_vulkan_probe LANGUAGES CXX)

add_executable(wallet-vulkan-probe vulkan_probe.cpp)
target_compile_features(wallet-vulkan-probe PRIVATE cxx_std_17)
target_compile_options(wallet-vulkan-probe PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wpedantic)

find_library(vulkan_library vulkan REQUIRED)
target_link_libraries(wallet-vulkan-probe PRIVATE "${vulkan_library}")

add_executable(wallet-vulkan-radix13-benchmark vulkan_radix13_benchmark.cpp)
target_compile_features(wallet-vulkan-radix13-benchmark PRIVATE cxx_std_17)
target_compile_options(wallet-vulkan-radix13-benchmark PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wpedantic)
target_link_libraries(wallet-vulkan-radix13-benchmark PRIVATE "${vulkan_library}")

add_executable(wallet-vulkan-radix2625-benchmark vulkan_radix13_benchmark.cpp)
target_compile_features(wallet-vulkan-radix2625-benchmark PRIVATE cxx_std_17)
target_compile_options(wallet-vulkan-radix2625-benchmark PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wpedantic)
target_link_libraries(wallet-vulkan-radix2625-benchmark PRIVATE "${vulkan_library}")
