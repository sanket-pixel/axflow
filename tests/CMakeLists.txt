include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# path to fixtures — passed as compile definition to all test targets
set(AXFLOW_TEST_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# helper to reduce boilerplate per test target
function(axflow_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE axflow::axflow GTest::gtest_main)
    target_compile_definitions(${target} PRIVATE
            AXFLOW_TEST_FIXTURES="${AXFLOW_TEST_FIXTURES}")
    gtest_discover_tests(${target})
endfunction()

# unit tests
axflow_add_test(test_axflow_config unit/config/test_axflow_config.cpp)

# integration tests — only with -DAXFLOW_BUILD_INTEGRATION_TESTS=ON
if (AXFLOW_BUILD_INTEGRATION_TESTS)
    axflow_add_test(test_yolov8_end_to_end integration/test_yolov8_end_to_end.cpp)
endif ()