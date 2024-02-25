#include "ppearls.h"

#define MAXWORDS 1024

int k = 2;
char inputChars[5*MAXWORDS];
char* words[MAXWORDS];
int wordCount = 0;

int next[MAXWORDS];
int bins[NHASH];

uint32_t hashByFirstNwords(char* p, int n)
{
	uint32_t h = 0;
	for(; n>0; p++)	
	{
		h = MULT*h + *p;
		if(*p == 0)
			--n;
	}
	return h%NHASH;
}

uint32_t hash(char* p)
{
	return hashByFirstNwords(p, k);
}

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

void printFirstNWords(int n)
{
	for(int i=0; i<wordCount; i++)
	{
		printf("words[%d]: ", i);
		for(int j=0; j<n; j++) printf("%s ", words[i+j]);		
		printf("\n");
	}
}

char* skipNwords(char* p, int n)
{
	for(; n>0; p++)
	{
		if(*p == 0)
			--n;
	}
	return p;
}

int main()
{
	srand((uint32_t)time(NULL));	
	init();	
	//printFirstNWords(1);

	for(int i=0; i<NHASH; i++)
		bins[i] = -1;

	for(int i=0; i<=wordCount-k; i++)
	{
		uint32_t j = hash(words[i]);
		next[i] = bins[j];
		bins[j] = i;	
	}
	
	char* p = NULL;
	char* phrase = inputChars;	
	for(int loop = 100; loop > 0; loop--)
	{
		int i = 0;
		for(int j=bins[hash(phrase)]; j>=0; j=next[j])
		{
			if((wordncmp(phrase, words[j]) == 0)
				&& (rand() % (i+1) == 0))
				p = words[j];
		}	
		phrase = skipNwords(p, 1);
		if(strlen(skipNwords(phrase, k-1)) == 0)
			break;		
		printf("%s ", skipNwords(phrase, k-1));
	}
	printf("\n");
	return 0;
}

