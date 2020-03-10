#include "apue.h"

int main()
{
    umask(0113);

    pid_t pid = fork();
    if(pid == -1)
    {
        err_sys("fork error");
    }
    else if(pid == 0)
    { // child
        int fd = open("test", O_CREAT|O_RDWR, S_IRWXU|S_IRWXG|S_IRWXO);
        if(fd == -1)
        {
            err_sys("open error");
        }
        close(fd);
    }
    else
    { // parent
        wait(NULL);

        struct stat statbuf;
        if(stat("test", &statbuf) != 0)
        {
            err_sys("stat error");
        }

        printf("%o\n", statbuf.st_mode);
    }
}

