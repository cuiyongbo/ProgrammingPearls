#include "apue.h"

#define LOCKFILE "/var/run/my_daemon.pid"
#define LOCKMONE (S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)

int already_running()
{
    int fd = open(LOCKFILE, O_RDWR|O_CREAT, LOCKMONE);
    if(fd < 0)
    {
        syslog(LOG_ERR, "can't open %s: %m", LOCKFILE);
        exit(1);
    }

    if(lockf(fd, F_TLOCK, 0) < 0)
    {
        if(errno == EACCES || errno == EAGAIN)
        {
            close(fd);
            exit(1);
        }
        syslog(LOG_ERR, "can't lock %s: %m", LOCKFILE);
        exit(1);
    }
    ftruncate(fd, 0);
    char buf[32];
    sprintf(buf, "%ld\n", (long)getpid());
    write(fd, buf, strlen(buf)+1);
    return 0;
}