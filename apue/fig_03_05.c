#include "apue.h"

int main()
{
	int n;
	char buf[BUFSIZ];
	while((n=read(STDIN_FILENO, buf, BUFSIZ)) > 0)
	{
		if(write(STDOUT_FILENO, buf, n) != n)
			err_sys("write error");
	}

	if(n<0)
		err_sys("read error");
	exit(0);
}

