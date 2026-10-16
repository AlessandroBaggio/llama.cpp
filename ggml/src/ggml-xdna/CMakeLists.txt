ggml_add_backend_library(ggml-xdna
    ggml-xdna.cpp
    ggml-xdna-sha256.c
    ggml-xdna-sha256.h
    ggml-xdna-rotate-bits.h
    ../../include/ggml-xdna.h
)

target_compile_features(
    ggml-xdna
    PRIVATE
    cxx_std_17
)

if(DEFINED ENV{XRT_ROOT} AND NOT "$ENV{XRT_ROOT}" STREQUAL "")
    set(_XRT_DEFAULT "$ENV{XRT_ROOT}")
else()
    set(_XRT_DEFAULT "C:/Xilinx/XRT")
endif()

set(
    XRT_ROOT
    "${_XRT_DEFAULT}"
    CACHE PATH
    "XRT Windows SDK installation root"
)

unset(_XRT_DEFAULT)

set(XRT_INCLUDE_DIR "${XRT_ROOT}/include")
set(XRT_COREUTIL_LIB "${XRT_ROOT}/lib/xrt_coreutil.lib")

if(NOT EXISTS "${XRT_INCLUDE_DIR}/xrt/xrt_device.h")
    message(FATAL_ERROR
        "XRT device header not found: ${XRT_INCLUDE_DIR}/xrt/xrt_device.h\n"
        "Install the XRT Windows SDK or configure -DXRT_ROOT=<path>."
    )
endif()

if(NOT EXISTS "${XRT_COREUTIL_LIB}")
    message(FATAL_ERROR
        "XRT core utility library not found: ${XRT_COREUTIL_LIB}\n"
        "Install the XRT Windows SDK or configure -DXRT_ROOT=<path>."
    )
endif()

target_include_directories(
    ggml-xdna
    PRIVATE
    "${XRT_INCLUDE_DIR}"
)

target_link_libraries(
    ggml-xdna
    PRIVATE
    "${XRT_COREUTIL_LIB}"
)

if(MSVC)
    target_compile_options(
        ggml-xdna
        PRIVATE
        /Zc:__cplusplus
    )
endif()

message(STATUS "XDNA XRT_ROOT: ${XRT_ROOT}")
message(STATUS "XDNA XRT include: ${XRT_INCLUDE_DIR}")
message(STATUS "XDNA XRT library: ${XRT_COREUTIL_LIB}")

option(GGML_XDNA_BUILD_DIAGNOSTICS "Build XDNA replay and capture diagnostics" OFF)
if(GGML_XDNA_BUILD_DIAGNOSTICS)
    if(GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_XDNA_BUILD_DIAGNOSTICS requires static XDNA backend linkage for its private replay entrypoint")
    endif()
    target_sources(ggml-xdna PRIVATE ggml-xdna-diagnostics.cpp)
    target_sources(ggml-xdna PRIVATE ggml-xdna-causal-payload-capture.cpp)
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)
    target_compile_definitions(ggml-base PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)
    add_executable(xdna-capture-session-selftest
        ../../../tools/xdna-capture-session-selftest.cpp
        ggml-xdna-sha256.c)
    target_compile_features(xdna-capture-session-selftest PRIVATE cxx_std_17)
    target_include_directories(ggml-xdna PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(xdna-causal-payload-capture-selftest
        ../../../tools/xdna-causal-payload-capture-selftest.cpp
        ggml-xdna-causal-payload-capture.cpp
        ggml-xdna-sha256.c)
    target_compile_features(xdna-causal-payload-capture-selftest PRIVATE cxx_std_17)
    target_include_directories(xdna-causal-payload-capture-selftest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor)

    add_executable(xdna-capture-replay
        ../../../tools/xdna-capture-replay.cpp
        ggml-xdna-sha256.c)
    target_compile_features(xdna-capture-replay PRIVATE cxx_std_17)
    target_include_directories(xdna-capture-replay PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(xdna-capture-replay PRIVATE ggml-base ggml ggml-xdna)
    add_library(xdna-causal-runtime SHARED
        ../../../tools/xdna-causal-runtime.cpp)
    target_compile_features(xdna-causal-runtime PRIVATE cxx_std_17)
    target_compile_definitions(xdna-causal-runtime PRIVATE GGML_XDNA_BUILD_DIAGNOSTICS)
    target_include_directories(xdna-causal-runtime PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(xdna-causal-runtime PRIVATE ggml-base ggml ggml-xdna)

    add_executable(xdna-causal-executor
        ../../../tools/xdna-causal-executor.cpp
        ggml-xdna-sha256.c)
    target_compile_features(xdna-causal-executor PRIVATE cxx_std_17)
    target_include_directories(xdna-causal-executor PRIVATE
        ../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools)

    add_executable(xdna-causal-execution-gate-selftest
        ../../../tools/xdna-causal-execution-gate-selftest.cpp)
    target_compile_features(xdna-causal-execution-gate-selftest PRIVATE cxx_std_17)
    target_include_directories(xdna-causal-execution-gate-selftest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
endif()

option(GGML_XDNA_RUNTIME_TRACE "Build opt-in production-linkable XDNA runtime tracing" OFF)
if(GGML_XDNA_RUNTIME_TRACE)
    if(GGML_XDNA_BUILD_DIAGNOSTICS)
        message(FATAL_ERROR "GGML_XDNA_RUNTIME_TRACE is trace-only and requires GGML_XDNA_BUILD_DIAGNOSTICS=OFF")
    endif()
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_RUNTIME_TRACE)
endif()
