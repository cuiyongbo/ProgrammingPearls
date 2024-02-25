#include "apue.h"
#include <sys/shm.h>

#define ARRAY_SIZE 4000
#define MALLOC_SIZE 10000
#define SHM_SIZE 10000
#define SHM_MODE 0600

char array[ARRAY_SIZE]; 
int x = 1234;

int main()
{
	printf("data seg x at %p\n", (void*)&x);
	printf("bss array[] from %p to %p\n", (void*)&array[0], (void*)&array[ARRAY_SIZE]);
	int shmid;
	printf("stack around shmid %p\n", (void*)&shmid);
	int au;
	au = 2;
	printf("stack around au %p\n", (void*)&au);

	char* ptr = (char*)malloc(MALLOC_SIZE);
	if(ptr == NULL)
		err_sys("malloc error");
	printf("heap malloced from %p to %p\n", (void*)ptr, (void*)(ptr+MALLOC_SIZE));
	
	shmid = shmget(IPC_PRIVATE, SHM_SIZE, SHM_MODE);
	if(shmid < 0)
		err_sys("shmget error");
	char* shmptr = shmat(shmid, 0, 0);
	if(shmptr == (void*)-1)
		err_sys("shmat error");
	printf("shared memory attached from %p to %p\n", (void*)shmptr, (void*)(shmptr+SHM_SIZE));
	if(shmctl(shmid, IPC_RMID, 0) < 0)
		err_sys("shmctl error");
	exit(0);
}
