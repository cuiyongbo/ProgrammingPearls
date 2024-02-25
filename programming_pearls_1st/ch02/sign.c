#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WORDMAX  100

int charcmp(const void* x, const void* y) 
{
	return *(char*)x - *(char*)y;
}

int main(void)
{
	char word[WORDMAX];
	char sig[WORDMAX];

	while(scanf("%s", word) != EOF)
	{
		strcpy(sig, word);
		qsort(sig, strlen(sig), sizeof(char), charcmp);
		printf("%s %s\n", sig, word);
	}
	return 0;
}

