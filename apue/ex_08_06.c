#include "apue.h"

int main()
{
    pid_t pid = fork();
    if (pid < 0)
    {
        err_sys("fork error");
        /* code */
    }
    else if(pid == 0)
    {
        printf("Child is going down\n");
        exit(0);
    }
    else
    {
        printf("Parent does not wait child\n");
        char buf[32];
        sprintf(buf, "ps %u", (unsigned int)pid);
        system(buf);
    }
    return 0;
}
