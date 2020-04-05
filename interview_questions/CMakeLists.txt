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

SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    message(STATUS "Configuring ${PROJECT_NAME} in debug mode")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Configuring ${PROJECT_NAME} in release mode")
else()
    message(STATUS "Unrecognized build type - will use cmake defaults")
endif()

# message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_executable(graph_evaluate_division graph_evaluate_division.cpp ${PUB_UTIL_FILES})
target_link_libraries(graph_evaluate_division)

add_executable(graph_redundant_connection graph_redundant_connection.cpp ${PUB_UTIL_FILES})
target_link_libraries(graph_redundant_connection)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)