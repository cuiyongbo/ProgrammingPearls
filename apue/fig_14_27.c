#include "apue.h"
#include <sys/mman.h>
#include <sys/stat.h>

#define COPYINCR (1024*1024*1024)

int main(int argc, char* argv[])
{
	if(argc != 3)
		err_quit("Usage: %s src dest", argv[0]);

	int fdin = open(argv[1], O_RDONLY);
	if(fdin < 0)
		err_sys("open(%s) error", argv[1]);

	int fdout = open(argv[2], O_RDWR|O_CREAT|O_TRUNC, FILE_MODE);
	if(fdout < 0)
		err_sys("open(%s) error", argv[2]);

	struct stat sbuf;
	if(fstat(fdin, &sbuf) < 0)
		err_sys("fstat error");

	if(ftruncate(fdout, sbuf.st_size) < 0)
		err_sys("ftruncate error");

	size_t copysz;
	off_t fsz = 0;
	while(fsz < sbuf.st_size)
	{
		if(sbuf.st_size - fsz > COPYINCR)
			copysz = COPYINCR;
		else
			copysz = sbuf.st_size - fsz;

		void* src = mmap(0, copysz, PROT_READ, MAP_SHARED, fdin, fsz);
		if(src == MAP_FAILED)
			err_sys("mmap error for input");

		void* dst = mmap(0, copysz, PROT_READ|PROT_WRITE, MAP_SHARED, fdout, fsz);
		if(dst == MAP_FAILED)
			err_sys("mmap error for output");

		memcpy(dst, src, copysz);
		munmap(src, copysz);
		munmap(dst, copysz);
		fsz += copysz;
	}
	exit(0);
}

