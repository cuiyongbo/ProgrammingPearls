#include "apue.h"

int main(int argc, char* argv[])
{
	for(int i=0; i<argc; ++i)
	{
		printf("argv[%d]: %s\n", i, argv[i]);
	}

	//extern char** environ;
	//for(char** p = environ; *p != NULL; ++p)
	//{
	//	printf("%s\n", *p);
	//}

	return 0;
}

