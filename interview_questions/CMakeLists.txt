cmake_minimum_required(VERSION 3.0.0)
project(scaffold VERSION 0.1.0)

include(CTest)
enable_testing()

set(PUB_UTIL_FILES util.cpp log.cpp disjoint_set.cpp)

include_directories()
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

add_executable(binary_search_search_insert_pos binary_search_search_insert_pos.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_insert_pos PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_insert_pos)

add_executable(binary_search_search_in_rotated_sorted_array binary_search_search_in_rotated_sorted_array.cpp ${PUB_UTIL_FILES})
target_compile_features(binary_search_search_in_rotated_sorted_array PUBLIC cxx_auto_type)
target_link_libraries(binary_search_search_in_rotated_sorted_array)

add_executable(graph_town_judge graph_town_judge.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_town_judge PUBLIC cxx_auto_type)
target_link_libraries(graph_town_judge)

add_executable(graph_network_delay_time graph_network_delay_time.cpp ${PUB_UTIL_FILES})
target_compile_features(graph_network_delay_time PUBLIC cxx_auto_type)
target_link_libraries(graph_network_delay_time)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
