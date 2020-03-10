#include "open.h"

int main(int argc, char* argv[])
{
	int fd, n;
	char buf[BUFSIZ];
	char line[BUFSIZ];
	while(fgets(line, BUFSIZ, stdin) != NULL)
	{
		if(line[strlen(line)-1] == '\n')
			line[strlen(line)-1] = 0; // replace newline with null

		if((fd = csopen(line, O_RDONLY)) < 0) continue;

		while((n=read(fd, buf, BUFSIZ)) > 0)
		{
			if(write(STDOUT_FILENO, buf, n) != n)
				err_sys("write error");
		}
		
		if(n<0) err_sys("read error");		
	
		close(fd);
	}
	return 0;
}

