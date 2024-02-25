cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)

project(protobuf_scaffold
	DESCRIPTION "Protocol Buffer Scaffold"
	VERSION 0.0.1
)

MESSAGE(${CMAKE_CURRENT_BINARY_DIR})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#file(GLOB_RECURSE PROTO_IDLS *.proto)
#MESSAGE(${PROTO_IDLS})
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_IDLS})
#add_executable(hello hello.cpp ${PROTO_SRCS} ${PROTO_HDRS})
#target_compile_features(hello PUBLIC cxx_auto_type)
#target_link_libraries(hello ${Protobuf_LIBRARIES})
#add_executable(add_person add_person.cpp ${PROTO_SRCS} ${PROTO_HDRS})
#target_compile_features(add_person PUBLIC cxx_auto_type)
#target_link_libraries(add_person ${Protobuf_LIBRARIES})
#add_executable(list_people list_people.cpp ${PROTO_SRCS} ${PROTO_HDRS})
#target_compile_features(list_people PUBLIC cxx_auto_type)
#target_link_libraries(list_people ${Protobuf_LIBRARIES})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/hello.proto)
add_executable(hello hello.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_compile_features(hello PUBLIC cxx_auto_type)
target_link_libraries(hello ${Protobuf_LIBRARIES})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/addressbook.proto)
add_executable(add_person add_person.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_compile_features(add_person PUBLIC cxx_auto_type)
target_link_libraries(add_person ${Protobuf_LIBRARIES})
add_executable(list_people list_people.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_compile_features(list_people PUBLIC cxx_auto_type)
target_link_libraries(list_people ${Protobuf_LIBRARIES})