#include "opend.h"

int buf_args(char* buf, int (*openfunc)(int, char**))
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
	return openfunc(argc, argv);
}

