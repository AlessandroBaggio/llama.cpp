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