#include "apue.h"

/*
natsume@ubuntu:apue $ cat ~/test_interpreter 
#!/mnt/hgfs/scaffold/ProgrammingPearls/apue/hello ttt

natsume@ubuntu:apue $ cat test.c
#include <stdio.h>

int main(int argc, char *argv[])
{
    for(int i=0; i<argc; i++)
        printf("%s ", argv[i]);
    printf("\n");
    return 0;
}
*/

int main()
{
	pid_t pid = fork();
	if(pid < 0)
	{
		err_sys("fork");
	}
	else if(pid == 0)
	{
		if(execlp("test_interpreter", "testinterp", "arg1", (char*)0) < 0)
		// if(execl("/home/natsume/test_interpreter", "test_interpreter", "arg1", (char*)0) < 0)
			err_sys("execl");
	}

	if(waitpid(pid, NULL, 0) < 0)
		err_sys("waitpid");

	exit(EXIT_SUCCESS);
}

