cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ptMgrad)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PTMGRAD_ROOT ${PROJECT_SOURCE_DIR})
set(PTMGRAD_SRCS_DIR "${PTMGRAD_ROOT}/src/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(PTMGRAD_SRCS)
list(APPEND PTMGRAD_SRCS
  ${PTMGRAD_SRCS_DIR}/engine.cpp
)

add_library(${PROJECT_NAME} SHARED ${PTMGRAD_SRCS})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wno-unused-function)

target_include_directories(${PROJECT_NAME} PUBLIC ${PTMGRAD_SRCS_DIR})

set(PTMGRAD_TESTS)
list(APPEND PTMGRAD_TESTS
  ${PTMGRAD_SRCS_DIR}/test/main.cpp
)
