#include "unp.h"
#include <syslog.h>

extern int  daemon_proc;    /* defined in error.c */

int daemon_init(const char* pname, int facility)
{
    pid_t pid = Fork();
    if(pid != 0)
    {
        // parent terminates
        exit(0);
    }

    // create session and set process group ID
    // On return calling process becomes the session leader of the new session,
    // the process group leader of a new process group and has no controlling terminal
    if(setsid() < 0)
        return -1;

    Signal(SIGHUP, SIG_IGN);
    Signal(SIGCHLD, SIG_IGN);

    // Only a session leader may acquire a terminal as its controlling terminal
    // ensure the double-forked process won't get a controlling terminal
    pid = Fork();
    if(pid != 0)
    {
        // child 1 terminates
        exit(0);
    }

    daemon_proc = 1;

    umask(0); // Set new file permissions

    chdir("/"); // change working directory

    // close all file descriptors
    long fdCount = sysconf(_SC_OPEN_MAX);
    for(int i=0; i<fdCount; ++i)
        close(i);

    // redirect stdin, stdout, stderr
    open("/dev/null", O_RDONLY);
    open("/dev/null", O_RDWR);
    open("/dev/null", O_RDWR);

    openlog(pname, LOG_PID, facility);

    return 0;
}

void daemon_inetd(const char *pname, int facility)
{
    daemon_proc = 1;        /* for our err_XXX() functions */
    openlog(pname, LOG_PID, facility);
}
