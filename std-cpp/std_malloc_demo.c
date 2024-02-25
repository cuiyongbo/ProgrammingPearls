#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>

// use ltrace -S ./a.out

int main()
{
    if(mallopt(M_TRIM_THRESHOLD, 128*1024) != 1)
    {
        fprintf(stderr, "mallopt(M_TRIM_THRESHOLD) error");
        exit(EXIT_FAILURE);
    }

    if(mallopt(M_MMAP_THRESHOLD, 10240*1024) != 1)
    {
        fprintf(stderr, "mallopt(M_MMAP_THRESHOLD) error");
        exit(EXIT_FAILURE);
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

    exit(EXIT_SUCCESS);
}
