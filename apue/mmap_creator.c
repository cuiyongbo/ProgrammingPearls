#include "apue.h"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <signal.h>

static int g_goToExit = 0;
void terminationHandler(int signo)
{
	g_goToExit = 1;
}

int main(int argc, char* argv[])
{
	if(argc != 2)
		err_quit("Usage: %s size", argv[0]);

	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	act.sa_handler = terminationHandler;
	if(sigaction(SIGTERM, &act, NULL) < 0)
		err_sys("sigaction(SIGTERM) error");

	const char* name = "/test-shm";
	int fd = shm_open(name, O_RDWR|O_CREAT|O_EXCL, S_IRWXU|S_IRWXG);
	if(fd < 0)
		err_sys("shm_open(%s) error", name);

	if(ftruncate(fd, atoi(argv[1])) < 0)
		err_sys("ftruncate failed");
	
	while(!g_goToExit)
	{
		sleep(5);
	}

	if(shm_unlink(name) < 0)
		err_sys("shm_unlink(%s) failed", name);

	return 0;
}

