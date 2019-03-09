cmake_minimum_required (VERSION 2.8.11)
set (PROC_OUTPUT "path_finder")
project (${PROC_OUTPUT})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/_build/lib)

#message(${PROJECT_BINARY_DIR})

set(PROJECT_SRC_DIR ${PROJECT_BINARY_DIR}/..)

add_definitions(-Wall -Wextra -g -std=c++11)

include_directories(${PROJECT_SRC_DIR}/alg)
include_directories(${PROJECT_SRC_DIR}/data_struct)
include_directories(${PROJECT_SRC_DIR}/util)
include_directories(${PROJECT_SRC_DIR}/src)

#link_directories(path/to/common_library)

file(GLOB_RECURSE SRC "${PROJECT_SRC_DIR}/src/*.cpp" "${PROJECT_SRC_DIR}/data_struct/*.cpp" "${PROJECT_SRC_DIR}/util/*.cpp" "${PROJECT_SRC_DIR}/alg/*.cpp")

# message(${SRC}) # print source files

# exclude certain file from build
list(REMOVE_ITEM SRC "${PROJECT_SRC_DIR}/util/implementation.cpp")

add_executable(${PROC_OUTPUT} ${SRC})

# link libraries
#target_link_libraries(${PROC_OUTPUT})
