#include "apue.h"

static void my_exit1(void)
{
	printf("my_exit1\n");
}

static void my_exit2(void)
{
	printf("my_exit2\n");
}

int main()
{
	if(atexit(my_exit2) != 0)
		err_sys("Failed to register my_exit2");

	if(atexit(my_exit1) != 0)
		err_sys("Failed to register my_exit1");
	if(atexit(my_exit1) != 0)
		err_sys("Failed to register my_exit1");

	printf("main is done\n");

	return 0;
}
