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
	sort();
	//printFirstNWords(k);
	
	char* p = NULL;
	char* phrase = inputChars;	
	for(int loop = 100; loop > 0; loop--)
	{
		int l = -1;
		int u = wordCount;
		while(l+1 != u)
		{
			int m = (l+u)/2;
			if(wordncmp(words[m], phrase) < 0)
				l = m;
			else
				u = m;
		}

		for(int i=0; wordncmp(phrase, words[u+i])==0; i++)
		{
			if(rand() % (i+1) == 0)
				p = words[u+i];
		}	
		phrase = skipNwords(p, 1);
		if(strlen(skipNwords(phrase, k-1)) == 0)
			break;		
		printf("%s ", skipNwords(phrase, k-1));
	}
	printf("\n");
	return 0;
}

