cmake_minimum_required(VERSION 3.0.0)
project(shalashara VERSION 0.1.0)

file(GLOB SRC main.cpp foo.cpp bar.cpp)
add_executable(${PROJECT_NAME} ${SRC})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_auto_type)
target_link_libraries(${PROJECT_NAME})
