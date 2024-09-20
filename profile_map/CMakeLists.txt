cmake_minimum_required(VERSION 3.0.0)
project(libtbb_scaffold VERSION 0.1.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(TBB_DIR "/path/to/oneTBB/")
set(TBB_HEADER_DIR ${TBB_DIR}/include)
set(TBB_LIB_DIR ${TBB_DIR}/lib)

set(UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../leetcode-scaffold/util")

include_directories (
    ${TBB_HEADER_DIR}
    ${UTIL_DIR}
    "/path/to/sparsehash/src"
)

link_directories (
    ${TBB_LIB_DIR}
)

include(CTest)
enable_testing()

add_executable(profile_map_op src/profile_map_op.cpp)
target_compile_features(profile_map_op PUBLIC cxx_auto_type)
target_link_libraries(profile_map_op tbb pthread)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
