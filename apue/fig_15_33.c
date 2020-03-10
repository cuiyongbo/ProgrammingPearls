#include "apue.h"
#include <fcntl.h>
#include <sys/mman.h>

#define NLOOPS 10
#define SIZE sizeof(long)

static int update(long* ptr)
{
	return ((*ptr)++);
}

int main()
{
	int fd = open("/dev/zero", O_RDWR);
	if(fd < 0)
		err_sys("open error");
	void* area = mmap(0, SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if(area == MAP_FAILED)
		err_sys("mmap error");
	close(fd);

	TELL_WAIT();

	int counter;
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid > 0)
	{
		for(int i=0; i<NLOOPS; i+=2)
		{
			counter = update((long*)area);
			err_return("parent: i: %d, counter: %d", i, counter);
			if(counter != i)
				err_quit("parent: expect %d, got %d", i, counter);

				TELL_CHILD(pid);
				WAIT_CHILD();
		}
	}
	else
	{
		for(int i=1; i<NLOOPS+1; i+=2)
		{
			WAIT_PARENT();
			counter = update((long*)area);
			err_return("child: i: %d, counter: %d", i, counter);
			if(counter != i)
				err_quit("child: expect %d, got %d", i, counter);
			TELL_PARENT(getppid());
		}
	}
	exit(0);
}

