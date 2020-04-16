cmake_minimum_required(VERSION 3.0.0)
project(hello-world VERSION 0.1.0)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(TBB_DIR "/mnt/c/VS-Demo/oneTBB/")
else()
    set(TBB_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(TBB_HEADER_DIR ${TBB_DIR}/include)
set(TBB_LIB_DIR ${TBB_DIR}/lib)

set(UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../interview_questions/util")

include_directories (
    ${TBB_HEADER_DIR}
    ${UTIL_DIR}
)

link_directories (
    ${TBB_LIB_DIR}
)

include(CTest)
enable_testing()

add_executable(parallel_for_demo_01 parallel_for_demo_01.cpp)
target_compile_features(parallel_for_demo_01 PUBLIC cxx_auto_type)
target_link_libraries(parallel_for_demo_01 tbb pthread)

add_executable(parallel_for_demo_02 parallel_for_demo_02.cpp)
target_compile_features(parallel_for_demo_02 PUBLIC cxx_auto_type)
target_link_libraries(parallel_for_demo_02 tbb pthread)

add_executable(concurrent_map_demo_01 concurrent_map_demo_01.cpp)
target_compile_features(concurrent_map_demo_01 PUBLIC cxx_auto_type)
target_link_libraries(concurrent_map_demo_01 tbb pthread)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
