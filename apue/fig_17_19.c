#include "open.h"
#include <sys/uio.h> // struct iovec

// Open the file by sending the name and oflag
// to the connection server and reading a file
// descriptor back.

int csopen(char* name, int oflag)
{
	static int fd[2] = {-1, -1};
	if(fd[0] < 0)
	{
		if(fd_pipe(fd) < 0)
		{
			err_ret("fd_pipe error");
			return -1;
		}
	
		pid_t pid = fork();
		if(pid < 0)
		{
			err_ret("fork error");
			return -1;
		}
		else if(pid == 0)
		{
			close(fd[0]);
			if(fd[1] != STDIN_FILENO &&
				dup2(fd[1], STDIN_FILENO) != STDIN_FILENO)
				err_sys("dup2 error to stdin");
	
			if(fd[1] != STDOUT_FILENO &&
				dup2(fd[1], STDOUT_FILENO) != STDOUT_FILENO)
				err_sys("dup2 error to stdout");
	
			if(execl("./opend", "opend", (char*)0) < 0)
				err_sys("execl error");
		}
		else
		{
			close(fd[1]);
		}
	}	

	char buf[10];
	sprintf(buf, " %d", oflag);
	struct iovec iov[3];
	iov[0].iov_base = CL_OPEN " ";
	iov[0].iov_len = strlen(CL_OPEN) + 1;
	iov[1].iov_base = name;
	iov[1].iov_len = strlen(name);
	iov[2].iov_base = buf;
	iov[2].iov_len = strlen(buf) + 1;
	ssize_t len = iov[0].iov_len + iov[1].iov_len + iov[2].iov_len;
	if(writev(fd[0], iov, 3) != len)
	{
		err_ret("writev error");
		return -1;
	}
	
	return recv_fd(fd[0], write);
}

