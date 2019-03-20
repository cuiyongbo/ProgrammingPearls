#include "apue.h"
#include <arpa/inet.h>

int main()
{
	FILE* fp = fopen("sample", "wb");
	if(fp == NULL)
		err_sys("fopen error");

	//fprintf(fp, "Hello world\n");
	fwrite("Hello world", 1,  strlen("Hello world"), fp);
	fclose(fp);	
	return 0;
}
