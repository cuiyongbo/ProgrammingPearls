#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#if !defined(WIN32)
#include <unistd.h>
#endif

#define MULT 31
#define NHASH 29989

typedef unsigned int uint32;

#define element_of_array(a) (sizeof(a)/sizeof(a[0]))

