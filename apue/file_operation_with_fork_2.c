#include "apue.h"

int main()
{
	char buff[] = "abcdefg";
	int fd = open("sample", O_CREAT|O_RDWR|O_APPEND, S_IRWXU);
	if(fd<0)
	{
		err_sys("open error");
	}

	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork error");
	}
	else if(pid == 0)
	{
		int ret = write(fd, buff, sizeof(buff));
		if(ret < 0)
		{
			err_sys("write error");
		}
		exit(0);
	}
	else
	{
		int ret = write(fd, buff, sizeof(buff));
		if(ret < 0)
		{
			err_sys("write error");
		}
		waitpid(pid, NULL, WNOHANG);
	}
	return 0;
}
