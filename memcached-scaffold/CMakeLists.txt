cmake_minimum_required(VERSION 3.0.0)
project(memcached-scaffold VERSION 0.1.0)

include_directories()
link_directories(
    /usr/local/lib
)

add_executable(scaffold main.cpp)
target_compile_features(scaffold PUBLIC cxx_auto_type)
target_link_libraries(scaffold memcached)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
