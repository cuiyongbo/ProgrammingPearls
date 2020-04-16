#pragma once
#include <iostream>

inline void set_stdout_no_buffering()
{
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);
}
