#include "apue.h"

int main(int argc, char *argv[])
{
	daemonize("MyDaemon");

	printf("Daemon [%d] started\n", (int)getpid());

	while(1)
	{
		sleep(5);
	}
    return 0;
}
