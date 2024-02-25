#include "apue.h"

static void charatatime(char* str)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	for(char* ptr=str; *ptr != 0; ptr++)
		putc(*ptr, stdout);
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
