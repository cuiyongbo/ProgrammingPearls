#include "apue.h"

int main()
{
	daemonize("getlog");
	FILE* fp = fopen("/tmp/getlog.out", "w");
	if(fp != NULL)
	{
		char* p = getlogin();
		if(p != NULL)
		{
			fprintf(fp, "login name: %s\n", p);
		}
		else
		{
			fprintf(fp, "no login name\n");
		}	
	}			
	exit(0);
}
