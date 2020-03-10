#include "opend.h"

int cli_args(int argc, char** argv)
{
	if(argc != 4 || strcmp(argv[0], CL_OPEN) != 0)
	{
		snprintf(errmsg, BUFSIZ, "Usage: pathname oflag\n");
		for(int i=0; i<argc; i++)
			sprintf(errmsg+strlen(errmsg), "%d: %s\n", i, argv[i]);
		return -1;
	}
	pathname = argv[1];
	oflag = atoi(argv[2]);
	return 0;
}

