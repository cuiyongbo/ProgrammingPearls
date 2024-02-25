#include "ppearls.h"

int k = 2;
char inputChars[5*1024];
char* words[1024];
int wordCount = 0;

void init()
{
	words[0] = inputChars;
	while(scanf("%s", words[wordCount]) != EOF)
	{
		words[wordCount + 1] = words[wordCount] + strlen(words[wordCount]) + 1;
		wordCount++;
	}
}

int wordncmp(char* p, char* q)
{
	int n = k;
	for(; *p == *q; p++, q++)
	{
		if(*p == 0 && --n == 0)
			return 0;
	}
	return *p - *q;
}

int sortcmp(const void* l, const void* r)
{
	return wordncmp(*(char**)l, *(char**)r);
}

void sort()
{
	for(int i=0; i<k; i++) words[wordCount][i] = 0;
	qsort(words, wordCount, sizeof(words[0]), sortcmp);
}

void printFirstNWords(int n)
{
	for(int i=0; i<wordCount; i++)
	{
		printf("words[%d]: ", i);
		for(int j=0; j<n; j++) printf("%s ", words[i+j]);		
		printf("\n");
	}
}

int main()
{
	init();	
	printFirstNWords(1);
	sort();
	printFirstNWords(k);

	return 0;
}

