# cuprate gRPC streaming sync library — only built when MONERO_ENABLE_GRPC_STREAM=ON.
#
# Compiles the cuprate_stream.proto file via protoc + grpc_cpp_plugin, then wraps
# the generated stub in a Pimpl C++ class (`cuprate_grpc_stream_client`) so the
# wallet2 translation unit does NOT need to include grpc++ headers — keeps the
# wallet's existing Boost-Asio / epee build clean of grpc/abseil ABI.
#
# Build deps:
#   macOS: brew install grpc
#   Linux: apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

# Monero pulls a vintage Protobuf in via Trezor device support, which conflicts
# with the modern protobuf::libupb targets that gRPC's CMake config exports.
# Use pkg-config instead — it just hands us include paths + library names, no
# imported-target machinery, no double-import errors.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPCPP   REQUIRED grpc++)
pkg_check_modules(GRPC     REQUIRED grpc)
pkg_check_modules(PROTOBUF REQUIRED protobuf)

if(NOT DEFINED GRPC_CPP_PLUGIN_PATH)
    find_program(GRPC_CPP_PLUGIN_PATH grpc_cpp_plugin REQUIRED)
endif()

set(PROTO_FILE  "${CMAKE_CURRENT_SOURCE_DIR}/cuprate_stream.proto")
set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(MAKE_DIRECTORY "${PROTO_OUT_DIR}")

set(GENERATED_PB
    "${PROTO_OUT_DIR}/cuprate_stream.pb.cc"
    "${PROTO_OUT_DIR}/cuprate_stream.pb.h"
    "${PROTO_OUT_DIR}/cuprate_stream.grpc.pb.cc"
    "${PROTO_OUT_DIR}/cuprate_stream.grpc.pb.h"
)

add_custom_command(
    OUTPUT ${GENERATED_PB}
    COMMAND protoc
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
        --cpp_out=${PROTO_OUT_DIR}
        --grpc_out=${PROTO_OUT_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_PATH}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "protoc + grpc_cpp_plugin: cuprate_stream.proto"
    VERBATIM
)

add_library(cuprate_grpc_stream STATIC
    grpc_block_stream_client.cpp
    ${GENERATED_PB}
)

# Modern abseil/grpc++ headers require C++17 (uses std::is_same_v etc.)
# Monero's default is C++14 — bump just our target so the rest is unaffected.
set_target_properties(cuprate_grpc_stream PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(cuprate_grpc_stream
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROTO_OUT_DIR}
            ${GRPCPP_INCLUDE_DIRS}
            ${GRPC_INCLUDE_DIRS}
            ${PROTOBUF_INCLUDE_DIRS}
)

target_link_directories(cuprate_grpc_stream
    PRIVATE
            ${GRPCPP_LIBRARY_DIRS}
            ${GRPC_LIBRARY_DIRS}
            ${PROTOBUF_LIBRARY_DIRS}
)

target_link_libraries(cuprate_grpc_stream
    PRIVATE
        ${GRPCPP_LIBRARIES}
        ${GRPC_LIBRARIES}
        ${PROTOBUF_LIBRARIES}
)

# Suppress noisy warnings in protobuf-generated code under our strict flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(${GENERATED_PB}
        PROPERTIES COMPILE_FLAGS
        "-Wno-unused-parameter -Wno-deprecated-declarations -Wno-unused-variable"
    )
endif()
