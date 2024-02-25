#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// run with ``MALLOC_CHECK_=1 ./a.out ``

int main(int argc, char const *argv[])
{
    if(argc > 1)
    {
        if(mallopt(M_CHECK_ACTION, atoi(argv[1])) != 1)
        {
            fprintf(stderr, "mallopt failed\n");
            exit(EXIT_FAILURE);
        }
    }

    char* p = (char*)malloc(100);
    if (p == NULL)
    {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    strcpy(p, "hello world!");

    free(p);
    printf("main: returned from the first free() call\n");

    free(p);
    printf("main: returned from the second free() call\n");

    return 0;
}


// use ltrace -S ./a.out

int test()
{
    if(mallopt(M_TRIM_THRESHOLD, 128*1024) != 1)
    {
        fprintf(stderr, "mallopt(M_TRIM_THRESHOLD) error");
        return 0;
    }

    if(mallopt(M_MMAP_THRESHOLD, 10240*1024) != 1)
    {
        fprintf(stderr, "mallopt(M_MMAP_THRESHOLD) error");
        return 0;
    }

    int block = 4096 * 1024 ;
    char * a = malloc(block);
    char * b = malloc(block);
    char * c = malloc(block);

    free(b);
    sleep(10 * 10);

    free(a);
    sleep(10 * 10);

    free(c);

    return 1;
}
