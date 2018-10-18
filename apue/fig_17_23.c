#include "opend.h"

int buf_args(char* buf, int (*openfunc)(int, char**))
{
	if(strtok(buf, WHITESPACE) == NULL)
		return -1;

	char* argv[MAXARGC];
	char* ptr;
	int argc = 0;
	while((ptr = strtok(NULL, WHITESPACE)) != NULL)
	{
		argv[argc] = ptr;
		if(++argc >= MAXARGC-1)
			return -1;
	}
	argv[argc] = (char*)NULL;
	return openfunc(argc, argv);
}

