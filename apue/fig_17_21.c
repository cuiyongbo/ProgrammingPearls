#include "opend.h"

char errmsg[BUFSIZ];
int oflag;
char* pathname;

int main()
{
	int nread;
	char buf[BUFSIZ];
	for(;;)
	{
		if((nread=read(STDIN_FILENO, buf, BUFSIZ)) < 0)
		{
			err_sys("read error on stream pipe");
		}
		else if (nread == 0)
			break;

		handle_request(buf, nread, STDOUT_FILENO);
	}
	return 0;
}

