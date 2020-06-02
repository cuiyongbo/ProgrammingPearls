#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

void print_wrapper(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args); // it has to be vprintf
    va_end(args);
}

int main()
{
    print_wrapper("%s, %s, %d, %d, %d\n", "hello", "nice to meet you", 0, 1, 2);
}
