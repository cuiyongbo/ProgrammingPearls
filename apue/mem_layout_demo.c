#include "apue.h"
#include <sys/mman.h>

int initializedDataSegVar = 99;
int bssVar[64];

int main()
{
	printf("Intialized data segment variable: %p\n", &initializedDataSegVar);
	printf("Uninitialized data segment variable (BSS): %p\n", &bssVar);

	int a = 4;
	int b = 5;
	printf("Stack variables: a<%p>, b<%p>\n", &a, &b);
	
	void* xx = malloc(64);
	*(int*)xx = 63;
	printf("Heap variables: xx<%p>\n", xx);	

	void* yy = malloc(64);
	*(int*)yy = 36;
	printf("Heap variables: yy<%p>\n", yy);	
	
	free(xx);
	free(yy);
	
	FILE* fp = fopen("mem_layout_demo.c", "r");
	if(fp == NULL)
		err_sys("fopen failed");

	void* addr = mmap(NULL, 256, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
	if(addr == MAP_FAILED)
		err_sys("mmap failed");

	printf("Memory-mapping variable: %p\n", addr);
	munmap(addr, 256);
	fclose(fp);
}
