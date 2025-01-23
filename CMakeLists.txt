cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ptMgrad)

add_subdirectory(third_party/googletest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PTMGRAD_ROOT ${PROJECT_SOURCE_DIR})
set(PTMGRAD_SRCS_DIR "${PTMGRAD_ROOT}/src")
set(PTMGRAD_TEST_DIR "${PTMGRAD_ROOT}/test")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(PTMGRAD_SRCS)
list(APPEND PTMGRAD_SRCS
    ${PTMGRAD_SRCS_DIR}/engine.cpp
    ${PTMGRAD_SRCS_DIR}/nn.cpp
)

add_library(${PROJECT_NAME} SHARED ${PTMGRAD_SRCS})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_include_directories(${PROJECT_NAME} PUBLIC ${PTMGRAD_SRCS_DIR})

# Create main executable
#add_executable(${PROJECT_NAME}_main
    #${PTMGRAD_ROOT}/demo.cpp
    #${PTMGRAD_ROOT}/example.cpp
#    ${PTMGRAD_TEST_DIR}/main.cpp
#)

#target_link_libraries(${PROJECT_NAME}_main
#    PRIVATE
#    ${PROJECT_NAME}
#    gtest
#    gtest_main
#)

# enable testing
enable_testing()

set(PTMGRAD_TESTS)
list(APPEND PTMGRAD_TESTS
    ${PTMGRAD_TEST_DIR}/test_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_complex_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_maths_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_complex_maths_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_comparison_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_unary_ops.cpp
    ${PTMGRAD_TEST_DIR}/test_activation_funcs.cpp
)

add_executable(${PROJECT_NAME}_tests ${PTMGRAD_TESTS})

target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
    ${PROJECT_NAME}
    gtest
    gtest_main
)

add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)