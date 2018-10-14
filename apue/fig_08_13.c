#include "apue.h"

static void charatatime(char* str)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	int c;
	for(char* ptr=str; (c=*ptr) != 0; ptr++)
		putc(c, stdout);
}

int main()
{
	TELL_WAIT();
	
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork");
	}
	else if(pid == 0)
	{
		WAIT_PARENT();
		charatatime("output from child\n");
	}
	else
	{
		charatatime("output from parent\n");
		TELL_CHILD(pid);
	}
	exit(0);
}

