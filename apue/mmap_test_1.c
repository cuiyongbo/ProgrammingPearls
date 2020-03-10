#include "apue.h"
#include <sys/mman.h>
#include <sys/stat.h>

int main(int argc, char* argv[])
{
	if(argc != 2)
		err_quit("Usage: %s file", argv[0]);

	int fd = open(argv[1], O_RDONLY);
	if(fd < 0)
		err_sys("open(%s) error", argv[1]);

	struct stat sbuf;
	if(fstat(fd, &sbuf) < 0)
		err_sys("fstat error");

	void* addr = mmap(0, sbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if(addr == MAP_FAILED)
		err_sys("mmap failed");
	
	if(mprotect(addr, sbuf.st_size, PROT_READ|PROT_WRITE) < 0)
		err_sys("mprotect failed");
	
	if(munmap(addr, sbuf.st_size)<0)
		err_sys("munmap error");
	
	return 0;
}

