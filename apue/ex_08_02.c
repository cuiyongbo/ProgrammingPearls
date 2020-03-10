#include "apue.h"

int globvar = 6;

void vfork_test()
{
    pid_t pid = vfork();
    if (pid < 0) 
    {
        err_sys("vfork error");
    } 
    else if (pid == 0) 
    {
        printf("child after vfork()\n");
        globvar++;
    }
    else
    {
        printf("parent after vfork()\n");
    }
}

int main(void)
{
    if ((setvbuf(stdout, NULL, _IONBF, 0)) != 0) {
        perror("setvbuf error");
        return -1;
    }

    printf("before vfork\n");
    
    vfork_test();    

    printf("pid = %ld, glob = %d\n", (long)getpid(), globvar);
    exit(0);
}