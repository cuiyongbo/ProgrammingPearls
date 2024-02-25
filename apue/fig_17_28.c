#include "opend.h"

int debug, oflag, clientCount;
char errmsg[BUFSIZ];
char* pathname;

int main(int argc, char* argv[])
{
	opterr = 0;
	int c;
	while((c=getopt(argc, argv, "d")) != EOF)
	{
		switch(c)
		{
		case 'd':
			debug = 1;
			break;
		case '?':
			err_quit("unrecognized option: -%c", optopt);
		}	
	}	
	
	if(debug == 0)
		daemonize("opend");

	loop();

	return 0;
}

