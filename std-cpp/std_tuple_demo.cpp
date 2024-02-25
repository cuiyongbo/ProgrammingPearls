#include <tuple>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdlib>

std::tuple<double, char, std::string> getStudent(int id)
{
	if(id == 0) return std::make_tuple(3.8, 'A', "Lisa");
	if(id == 1) return std::make_tuple(2.5, 'B', "Cherry");
	if(id == 2) return std::make_tuple(1.8, 'C', "Luo");
	throw std::invalid_argument("invalid id");
}

int main(int argc, char* argv[])
{
	auto student0 = getStudent(0);	
	std::cout << "ID: 0, "
			  << "GPA: " << std::get<0>(student0) << ", "
			  << "grade: " << std::get<1>(student0) << ", "
			  << "name: " << std::get<2>(student0) << "\n";

	std::cout << "tuple size: " 
				<< std::tuple_size<decltype(student0)>::value << '\n';

	if (argc != 2)
	{
		printf("Usage: %s id\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	double gpa;
	char grade;
	std::string name;
	try
	{
		std::tie(gpa, grade, name) = getStudent(atoi(argv[1]));
		std::cout << "ID: " << argv[1] << ", "
			  << "GPA: " << gpa << ", "
			  << "grade: " << grade << ", "
			  << "name: " << name << "\n";
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << '\n';
	}
}
