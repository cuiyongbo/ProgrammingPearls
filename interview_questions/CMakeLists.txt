cmake_minimum_required(VERSION 3.0.0)
project(scaffold VERSION 0.1.0)

# message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
# message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

include(CTest)
enable_testing()

file(GLOB_RECURSE PUB_UTIL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp")
# message(STATUS ${PUB_UTIL_FILES})

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

SET(CMAKE_CXX_FLAGS " -Wall")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    message(STATUS "Configuring ${PROJECT_NAME} in debug mode")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Configuring ${PROJECT_NAME} in release mode")
else()
    message(STATUS "Unrecognized build type - will use cmake defaults")
endif()

# message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# message("List of compile features: ${CMAKE_CXX_COMPILE_FEATURES}")

add_executable(graph_evaluate_division graph_evaluate_division.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_evaluate_division PUBLIC cxx_std_17)
target_link_libraries(graph_evaluate_division)

add_executable(graph_redundant_connection graph_redundant_connection.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_redundant_connection PUBLIC cxx_auto_type) # let cmake deduce the c++ standard to use
target_link_libraries(graph_redundant_connection)

add_executable(graph_town_judge graph_town_judge.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_town_judge PUBLIC cxx_auto_type)
target_link_libraries(graph_town_judge)

add_executable(graph_network_delay_time graph_network_delay_time.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_network_delay_time PUBLIC cxx_auto_type)
target_link_libraries(graph_network_delay_time)

add_executable(graph_shortest_path_visiting_all_nodes graph_shortest_path_visiting_all_nodes.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_shortest_path_visiting_all_nodes PUBLIC cxx_auto_type)
target_link_libraries(graph_shortest_path_visiting_all_nodes)

# BINARY SEARCH

add_executable(binary_search_search_insert_pos binary_search_search_insert_pos.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_insert_pos PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_insert_pos)

add_executable(binary_search_search_in_rotated_sorted_array binary_search_search_in_rotated_sorted_array.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_in_rotated_sorted_array PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_in_rotated_sorted_array)

add_executable(binary_search_matrix_search binary_search_matrix_search.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_matrix_search PUBLIC cxx_auto_type)
target_link_libraries(binary_search_matrix_search)

add_executable(binary_search_sqrt binary_search_sqrt.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_sqrt PUBLIC cxx_auto_type)
target_link_libraries(binary_search_sqrt)

add_executable(median_finder median_finder.cpp ${PUB_UTIL_FILES})
target_compile_features(median_finder PUBLIC cxx_auto_type)
target_link_libraries(median_finder)

add_executable(binary_search_kth_smallest_pair_dis binary_search_kth_smallest_pair_dis.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_kth_smallest_pair_dis PUBLIC cxx_auto_type)
target_link_libraries(binary_search_kth_smallest_pair_dis)

add_executable(backtrace_combination backtrace_combination.cpp ${PUB_UTIL_FILES})
target_compile_features(backtrace_combination PUBLIC cxx_auto_type)
target_link_libraries(backtrace_combination)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)