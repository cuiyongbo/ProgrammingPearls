#include "ppearls.h"

#define MULT 31
#define NHASH 29989

typedef struct HashNode* HashNodePtr;
typedef struct HashNode
{
	char* word;
	int count;
	HashNodePtr next;
}HashNode;
static HashNodePtr g_bin[NHASH];

uint32_t hash(char* p)
{
	uint32_t h = 0;
	for(; *p; p++)
	{
		h = MULT * h + *p;
	}	
	return h % NHASH;
}

void incword(char* s)
{
	uint32_t h = hash(s);
	for(HashNodePtr p = g_bin[h]; p != NULL; p = p->next)
	{
		if(strcmp(p->word, s) == 0)
		{
			p->count++;
			return;
		}
	}

	HashNodePtr p = (HashNodePtr)malloc(sizeof(HashNode));
	p->word = strdup(s);
	p->count = 1;
	p->next = g_bin[h];
	g_bin[h] = p;
}

void freeHashMap()
{
	for(int i=0; i<NHASH; i++)
	{
		for(HashNodePtr p = g_bin[i]; p != NULL; )
		{
			HashNodePtr q = p->next;
			free(p->word);
			free(p);
			p = q;	
		}
	}
}

int main()
{
	for(int i=0; i<NHASH; i++) g_bin[i] = NULL;
	char buf[BUFSIZ];
	while(scanf("%s", buf) != EOF)
		incword(buf);
	for(int i=0; i<NHASH; i++)
	{
		for(HashNodePtr p = g_bin[i]; p!= NULL; p = p->next)
			printf("%s, %d\n", p->word, p->count);
	}
	
	freeHashMap();

	return 0;
}
