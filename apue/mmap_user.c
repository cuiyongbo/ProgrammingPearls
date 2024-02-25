#include "apue.h"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <signal.h>

int main(int argc, char* argv[])
{
	if(argc != 2)
		err_quit("Usage: %s size", argv[0]);

	const char* name = "/test-shm";
	int fd = shm_open(name, O_RDWR, S_IRWXU|S_IRWXG);
	if(fd < 0)
		err_sys("shm_open(%s) error", name);

	size_t size = atoi(argv[1]);
	void* addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if(addr == MAP_FAILED)
		err_sys("mmap failed");

	unsigned char* bytes = (unsigned char*)addr;
	for(size_t i=0; i<size*2; i++)
		bytes[i] = 'b';

	bytes[2*size-1] = 0;

	printf("bytes: %s\n", bytes);

	if(munmap(addr, size) < 0)
		err_sys("munmap failed");

	return 0;
}

