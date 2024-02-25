cmake_minimum_required(VERSION 3.0.0)
project(memcached-scaffold VERSION 0.1.0)

include_directories()

link_directories (
    /usr/local/lib
)

set(UTIL random_data_generator.cpp print_callbacks.cpp memcached_action.cpp)

add_executable(basic_operations_demo_01 basic_operations_demo_01.cpp)
target_compile_features(basic_operations_demo_01 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_demo_01 memcached)

add_executable(basic_operations_demo_02 basic_operations_demo_02.cpp ${UTIL})
target_compile_features(basic_operations_demo_02 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_demo_02 memcached)

add_executable(basic_operations_demo_03 basic_operations_demo_03.cpp ${UTIL})
target_compile_features(basic_operations_demo_03 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_demo_03 memcached)

add_executable(basic_operations_demo_04 basic_operations_demo_04.cpp ${UTIL})
target_compile_features(basic_operations_demo_04 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_demo_04 memcached)

add_executable(basic_operations_demo_05 basic_operations_demo_05.cpp ${UTIL})
target_compile_features(basic_operations_demo_05 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_demo_05 memcached)

add_executable(basic_operations_with_key_demo_01 basic_operations_with_key_demo_01.cpp)
target_compile_features(basic_operations_with_key_demo_01 PUBLIC cxx_auto_type)
target_link_libraries(basic_operations_with_key_demo_01 memcached)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
