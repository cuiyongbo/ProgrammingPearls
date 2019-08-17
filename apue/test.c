#include "apue.h"

void handler(int signo)
{
    printf("errno: %d(%s)\n", errno, strerror(errno));
    printf("receive signal %d(%s)\n", signo, strsignal(signo));
}


int main(int argc, char *argv[])
{
    if(signal(SIGTERM, handler) == SIG_ERR)
    {
        err_sys("signal error");
    }

    printf("PID: %ld\n", getpid());

	while(1)
	{
		sleep(50);
	}
    return 0;
}
