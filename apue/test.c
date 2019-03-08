#include "apue.h"
#include <arpa/inet.h>

int main()
{
	FILE* fp = fopen("sample", "w");
	if(fp == NULL)
		err_sys("fopen error");

	fprintf(fp, "Hello world\n");
	fclose(fp);	
	return 0;
}
