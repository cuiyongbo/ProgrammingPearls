#include "apue.h"

int main()
{
	int n = 10;
	char buff[10];
	//int fd = open("sample", O_RDONLY);
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid == 0)
	{
		int fd = open("sample", O_RDONLY);
		int ret = read(fd, buff, n);
		if(ret < 0)
		{
			err_sys("read error");
		}
		buff[ret-1] = 0;
		printf("child: %s\n", buff);
		exit(0);
	}
	else
	{
		int fd = open("sample", O_RDONLY);
		int ret = read(fd, buff, n);
		if(ret < 0)
		{
			err_sys("read error");
		}
		buff[ret-1] = 0;
		printf("parent: %s\n", buff);
		waitpid(pid, NULL, WNOHANG);
	}
	return 0;
}
