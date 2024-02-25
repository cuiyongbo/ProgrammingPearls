#include "apue.h"

int main(int argc, char* argv[])
{
	if(argc < 2)
		err_quit("Usage: %s file1 file2...", argv[0]);

	struct stat buf;
	for(int i=1; i<argc; i++)
	{
		printf("%s: ", argv[i]);
		if(lstat(argv[i], &buf) < 0)
		{
			err_msg("lstat error");
			continue;
		}
		
		char* ptr = NULL;
		if(S_ISREG(buf.st_mode))
			ptr = "regular";
		else if(S_ISDIR(buf.st_mode))
			ptr = "directory";
		else if(S_ISCHR(buf.st_mode))
			ptr = "character special";
		else if(S_ISBLK(buf.st_mode))
			ptr = "block special";
		else if(S_ISFIFO(buf.st_mode))
			ptr = "fifo";
		else if(S_ISLNK(buf.st_mode))
			ptr = "symbolic link";
		else if(S_ISSOCK(buf.st_mode))
			ptr = "socket";
		else if(S_TYPEISMQ(&buf))
			ptr = "message queue";
		else if(S_TYPEISSEM(&buf))
			ptr = "semaphore";
		else if(S_TYPEISSHM(&buf))
			ptr = "shared memory";
		else
			ptr = "** unknown mode **";

		printf("%s\n", ptr);
		printf("file size: %d\n", (int)buf.st_size);
	}
	return 0;
}

