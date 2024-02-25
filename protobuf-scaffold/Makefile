.PHONY: all hello add_person_cpp list_people_cpp clean

all: hello add_person_cpp list_people_cpp

hello: hello.cpp
	c++ -std=c++11 hello.cpp hello.pb.cc -o hello -lprotobuf

add_person_cpp: add_person.cpp
	c++ -std=c++11 add_person.cpp addressbook.pb.cc -o add_person_cpp -lprotobuf

list_people_cpp: list_people.cpp
	c++ -std=c++11 list_people.cpp addressbook.pb.cc -o list_people_cpp -lprotobuf

clean:
	rm -f hello add_person_cpp list_people_cpp a.out *.o