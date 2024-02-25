#include "apue.h"

int main()
{
	if(signal(SIGABRT, SIG_IGN) < 0)
		err_sys("signal error");
	
	abort();
}
