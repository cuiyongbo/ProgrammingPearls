#pragma once
#include <iostream>
#include <string>

class TestLogger
{
public:
    TestLogger(std::string test_case_name)
        : m_case(test_case_name)
    {
        std::cout << m_case << " begin" << std::endl;
    }

    ~TestLogger()
    { 
        std::cout << m_case << " end" << std::endl;
    }

private:
    std::string m_case;
};
