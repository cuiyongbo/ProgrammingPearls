#include "apue.h"

sigset_t g_mask;

void* threadFunc(void* arg)
{
    while(1)
    {
        int signo;
        int err = sigwait(&g_mask, &signo);
        if(err != 0)
        {
            syslog(LOG_ERR, "sigwait failed: %s", strerror(err));
            exit(1);
        }
        switch(signo)
        {
        case SIGHUP:
            syslog(LOG_INFO, "Re-reading configuration");
            // reload configuration
            break;
        case SIGTERM:
            syslog(LOG_INFO, "got SIGTERM; exiting");
            break;
        default:
            syslog(LOG_INFO, "unexpected signal %d", signo);
            break;
        }
        return 0;
    }
}

char* g_log_file_name = "test_daemon.log"; 

void log_message(char* message)
{
    FILE* logfile = fopen(g_log_file_name, "a");
    if(!logfile) return;
    fprintf(logfile, "%s\n", message);
    fclose(logfile);
}

int main(int argc, char* argv[])
{
    char* cmd;
    if((cmd=strrchr(argv[0], '/')) == NULL)
    {
        cmd = argv[0];
    }
    else
    {
        cmd++;
    }

    daemonize(cmd);

	syslog(LOG_WARNING, "Daemon started by cyb");

    if(already_running())
    {
        syslog(LOG_ERR, "daemon already running");
        exit(1);
    }

    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if(sigaction(SIGHUP, &sa, NULL) < 0)
    {
        log_message("can't restore SIGHUP to default action");
    }

    sigfillset(&g_mask);
    int err = pthread_sigmask(SIG_BLOCK, &g_mask, NULL);
    if( err != 0)
    {
        log_message("SIG_BLOCK error");
    }

    pthread_t tid;
    err = pthread_create(&tid, NULL, threadFunc, 0);
    if(err != 0)
    {
        log_message("pthread_create error");
    }

    while(1)
    {
        sleep(10);
    }

    exit(0);
}
