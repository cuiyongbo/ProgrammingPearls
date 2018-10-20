#include "apue.h"

#define WHITESPACE " "
#define MAXARGC 50

int buf_args(char* buf)
{
	if(strtok(buf, WHITESPACE) == NULL)
		return -1;

	char* argv[MAXARGC];
	char* ptr;
	int argc = 0;
	argv[argc] = buf;
	while((ptr = strtok(NULL, WHITESPACE)) != NULL)
	{
		if(++argc >= MAXARGC-1)
			return -1;
		argv[argc] = ptr;
	}
	argv[++argc] = (char*)NULL;
	return 0;
}

int main()
{
	char str[] = "open sample 4";
	buf_args(str);
	return 0;
}

