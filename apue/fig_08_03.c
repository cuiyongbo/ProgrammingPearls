#include "apue.h"

int globvar = 6;

int main(void)
{
    int var = 88;
    printf("before vfork\n");
    
    pid_t   pid = vfork();
    if (pid < 0) 
    {
        err_sys("vfork error");
    } 
    else if (pid == 0) 
    {
        globvar++;
        var++;
        //_exit(0);
        fclose(stdout);
        exit(0);
    }

    int ret = printf("pid = %ld, glob = %d, var = %d\n", (long)getpid(), globvar, var);
    ret++;
    exit(0);
}