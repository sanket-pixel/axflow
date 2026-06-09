cmake_minimum_required(VERSION 3.18)
project(axflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# build options
option(AXFLOW_BUILD_EXAMPLES "Build example programs" ON)
option(AXFLOW_BUILD_SHARED "Build axflow as shared lib" OFF)

message(STATUS "──────────────────────────────────────────")
message(STATUS "  AxFlow ${PROJECT_VERSION}")
message(STATUS "  Examples : ${AXFLOW_BUILD_EXAMPLES}")
message(STATUS "──────────────────────────────────────────")

# ──────────────────────────────────────────────
# Voyager SDK root
# Priority: -DVOYAGER_SDK_ROOT > env var VOYAGER_SDK_ROOT
# ──────────────────────────────────────────────
if (NOT VOYAGER_SDK_ROOT)
    if (DEFINED ENV{VOYAGER_SDK_ROOT})
        set(VOYAGER_SDK_ROOT $ENV{VOYAGER_SDK_ROOT})
    endif ()
endif ()
if (VOYAGER_SDK_ROOT)
    message(STATUS "  Voyager SDK : ${VOYAGER_SDK_ROOT}")
else ()
    message(WARNING "VOYAGER_SDK_ROOT not set — ORT may not be found.\n"
            "Pass -DVOYAGER_SDK_ROOT=/path/to/voyager-sdk or export it.")
endif ()

# ──────────────────────────────────────────────
# Detect architecture → pick correct ORT subdir
# ──────────────────────────────────────────────
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_ARCH_DIR "onnxruntime-linux-aarch64-1.17.1")
else ()
    set(ORT_ARCH_DIR "onnxruntime-linux-x64-1.17.1")
endif ()
set(ORT_SDK_ROOT "${VOYAGER_SDK_ROOT}/operators/onnxruntime/${ORT_ARCH_DIR}")
message(STATUS "  ORT subdir  : ${ORT_SDK_ROOT}")

# ──────────────────────────────────────────────
# AxRuntime root — default /opt/axelera
# Override with -DAXRUNTIME_ROOT=...
# ──────────────────────────────────────────────
if (NOT AXRUNTIME_ROOT)
    set(AXRUNTIME_ROOT "/opt/axelera")
endif ()
message(STATUS "  AxRuntime   : ${AXRUNTIME_ROOT}")

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# onnxruntime
find_library(ORT_LIB
        NAMES onnxruntime
        HINTS ${ORT_SDK_ROOT}/lib
        /usr/lib
        /usr/local/lib)
find_path(ORT_INCLUDE
        NAMES onnxruntime_cxx_api.h
        HINTS ${ORT_SDK_ROOT}/include
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime)

if (NOT ORT_LIB OR NOT ORT_INCLUDE)
    message(FATAL_ERROR
            "ONNXRuntime not found.\n"
            "Expected under: ${ORT_SDK_ROOT}\n"
            "Set -DVOYAGER_SDK_ROOT= correctly.")
endif ()
message(STATUS "  ORT lib     : ${ORT_LIB}")
message(STATUS "  ORT include : ${ORT_INCLUDE}")

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}")

# axruntime
file(GLOB AXRUNTIME_VERSION_DIRS "${AXRUNTIME_ROOT}/runtime-*/include")
find_path(AXR_INCLUDE
        NAMES axruntime/axruntime.hpp
        HINTS ${AXRUNTIME_VERSION_DIRS}
        ${AXRUNTIME_ROOT}/include)

find_library(AXR_LIB
        NAMES axruntime
        HINTS ${AXRUNTIME_VERSION_DIRS}/../lib
        ${AXRUNTIME_ROOT}/lib)

if (NOT AXR_LIB OR NOT AXR_INCLUDE)
    message(FATAL_ERROR
            "AxRuntime not found.\n"
            "Expected under: ${AXRUNTIME_ROOT}\n"
            "Override with -DAXRUNTIME_ROOT=...")
endif ()
message(STATUS "  AXR lib     : ${AXR_LIB}")
message(STATUS "  AXR include : ${AXR_INCLUDE}")

add_library(axruntime::axruntime UNKNOWN IMPORTED)
set_target_properties(axruntime::axruntime PROPERTIES
        IMPORTED_LOCATION "${AXR_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${AXR_INCLUDE}")

# axflow library
set(AXFLOW_SOURCES
        # config
        src/config/config_base.cpp
        src/config/preprocessing_config.cpp
        src/config/inference_config.cpp
        src/config/postprocessing_config.cpp
        src/config/axflow_config.cpp

        # core
        src/device/device.cpp
        src/inference/inference.cpp
        src/preprocess/preprocessor.cpp
        src/postprocess/postprocessor.cpp

        # top-level orchestrator
        src/axflow.cpp

        # utils
        src/utils/dequantize.cpp
        src/utils/tensor_layout.cpp
        src/utils/quantize.cpp
)

if (AXFLOW_BUILD_SHARED)
    add_library(axflow SHARED ${AXFLOW_SOURCES})
else ()
    add_library(axflow STATIC ${AXFLOW_SOURCES})
endif ()

add_library(axflow::axflow ALIAS axflow)

target_include_directories(axflow
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(axflow
        PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp
        onnxruntime::onnxruntime
        axruntime::axruntime
)

# examples
if (AXFLOW_BUILD_EXAMPLES)
    set(EXAMPLE_TARGETS
            01_basic_inference
            02_yolov8_detection
            03_lightfc_tracking
            04_segmentation
            05_depth_estimation
            06_keypoint_matching
            07_pose_estimation
            08_parallel_models
    )
    foreach (EXAMPLE ${EXAMPLE_TARGETS})
        add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)
        target_link_libraries(${EXAMPLE} PRIVATE axflow::axflow)
    endforeach ()
endif ()

# install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS axflow
        EXPORT axflowTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/axflow
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT axflowTargets
        FILE axflowTargets.cmake
        NAMESPACE axflow::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axflow
)

configure_package_config_file(
        cmake/axflowConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/axflowConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axflow
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/axflowConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/axflowConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/axflowConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/axflow
)

option(AXFLOW_BUILD_TESTS "Build unit tests" ON)
option(AXFLOW_BUILD_INTEGRATION_TESTS "Build integration tests" OFF)

if (AXFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()