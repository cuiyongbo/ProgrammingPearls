#include "apue.h"

int main(int argc, char* argv[])
{
	if(argc != 2)
		err_quit("Usage: %s <descriptor#>", argv[0]);

	int val = fcntl(atoi(argv[1]), F_GETFL, 0);
	if(val < 0)
		err_sys("fcntl error for fd %s", argv[1]);

	switch(val & O_ACCMODE)
	{
	case O_RDONLY:
		printf("read only");
		break;
	case O_WRONLY:
		printf("write only");
		break;
	case O_RDWR:
		printf("read and write");
		break;
	default:
		err_dump("unknown access mode");
	}

	if(val & O_APPEND)
		printf(", append");
	if(val & O_NONBLOCK)
		printf(", nonblocking");
	if(val & O_SYNC)
		printf(", synchronous writes");
	
	putchar('\n');
	return 0;
}

