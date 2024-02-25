#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

int main()
{
    printf("Process: %ld, Limit cap: %ld\n", (long)getpid(), RLIM_INFINITY);

    struct rlimit limit;
    if(getrlimit(RLIMIT_AS, &limit) != 0)
    {
        perror("getrlimit(RLIMIT_AS) error");
        exit(1);
    }

    printf("Limit of process's virtual memory (address space) in bytes:\n");
    printf("soft limit: %ld, hard limit: %ld\n", limit.rlim_cur, limit.rlim_max);

    if(getrlimit(RLIMIT_RSS, &limit) != 0)
    {
        perror("getrlimit(RLIMIT_RSS) error");
        exit(1);
    }

    printf("Limit of process's RSS (the number of virtual pages resident in RAM):\n");
    printf("soft limit: %ld, hard limit: %ld\n", limit.rlim_cur, limit.rlim_max);

    if(getrlimit(RLIMIT_STACK, &limit) != 0)
    {
        perror("getrlimit(RLIMIT_STACK) error");
        exit(1);
    }

    printf("Limit of process's stack in bytes:\n");
    printf("soft limit: %ld, hard limit: %ld\n", limit.rlim_cur, limit.rlim_max);
}