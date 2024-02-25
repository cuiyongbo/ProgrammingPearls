#include "apue.h"
#include <syslog.h>

char* g_log_file_name = "test_daemon.log"; 
char* g_pid_file_name = "test_daemon.pid"; 

void log_message(char* message)
{
    FILE* logfile = fopen(g_log_file_name, "a");
    if(!logfile) return;
    fprintf(logfile, "%s\n", message);
    fclose(logfile);
}

static void go_daemonize()
{
    pid_t pid = fork();
    if (pid < 0) {
        err_sys("fork error");
    }

    /* Success: Let the parent terminate */
    // so the orphaned child starts worked in background
    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    /* On success: The child process becomes session leader */
    /*
        A process receives signals from the terminal that it is connected to, 
        and each process inherits its parent's controlling tty. A server should not 
        receive signals from the process that started it, so it must detach itself 
        from its controlling tty.

        start a new process group and session, so detaching the orphaned process
        from its controlling tty
    */
    if (setsid() < 0) {
        err_sys("setsid error");
    }

    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    // https://stackoverflow.com/questions/17954432/creating-a-daemon-in-linux
    // https://stackoverflow.com/questions/881388/what-is-the-reason-for-performing-a-double-fork-when-creating-a-daemon
    /* Fork off for the second time*/
    // ensure the double-forked process won't get a controlling tty
    pid = fork();
    if (pid < 0) {
        err_sys("fork error");
    }

    /* Success: Let the parent terminate */
    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    /* Set new file permissions */
    umask(0);

    /* Change the working directory to the root directory */
    /* or another appropriated directory */
	char* homeDir = getenv("HOME");
    if(chdir(homeDir) < 0)
    {
        err_sys("chdir error");
    }

    /* Close all open file descriptors */
    for (int fd = sysconf(_SC_OPEN_MAX); fd > 0; fd--) {
        close(fd);
    }

    stdin = fopen("/dev/null", "r");
    stdout = fopen("/dev/null", "w+");
    stderr = fopen("/dev/null", "w+");

    {
        // Most services require running only one copy of a server at a time. 
        // File locking method is a good solution for mutual exclusion. 
        // The first instance of the server locks the file so that other instances 
        // understand that an instance is already running. If server terminates lock 
        // will be automatically released so that a new instance can run.
        // Another bonus it would be more convinient to find the server's pid
        // with `cat daemon.pid` than `ps -xj | grep daemon`.

        int pid_fd = open(g_pid_file_name, O_RDWR|O_CREAT, 0640);
        if (pid_fd < 0) {
            log_message("open error");
            exit(EXIT_FAILURE);
        }

        if (lockf(pid_fd, F_TLOCK, 0) < 0) {
            log_message("lockf error");
            exit(EXIT_FAILURE);
        }

        char str[32];
        sprintf(str, "%d\n", getpid());
        write(pid_fd, str, strlen(str));
    }
}

int main()
{
    go_daemonize();

    openlog("testDaemon", LOG_PID, LOG_DAEMON);

    while (1)
    {
        //TODO: Insert daemon code here.
        syslog (LOG_NOTICE, "Test daemon started.");
        sleep (20);
        break;
    }

    syslog (LOG_NOTICE, "Test daemon terminated.");
    closelog();
    return 0;
}

