#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 1024

int comlen(char* p, char* q)
{
	int i=0;
	while(*p && (*p++ == *q++))
		i++;
	return i;
}

void naiveSearchMaxDuplicatedSubstring(char* s, int len)
{
	int maxi = 0;
	int maxj = 0;
	int maxlen = -1;
	for(int i=0; i<len; i++)
	{
		for(int j=i+1; j<len; j++)
		{
			int matchedLen = comlen(s+i, s+j);
			if(matchedLen > maxlen)
			{
				maxlen = matchedLen;
				maxi = i;
				maxj = j;
			}
		}
	}	

	printf("<%s>|<%.*s>\n", s, maxlen, s+maxi);
}

int pstrcmp(const void* p, const void* q)
{
	return strcmp(*(const char**)p, *(const char**)q);
}

int main()
{
	char c[MAXN];
	char* a[MAXN];
	int ch;
	int n = 0;
	while((ch=getchar()) != EOF)
	{
		a[n] = &c[n];
		c[n++] = ch;
	}
	c[n] = 0;

//	scanf("%s", c);
//	for(n=0; c[n] != 0; n++)
//	{
//		a[n] = c + n;
//	}

	//naiveSearchMaxDuplicatedSubstring(c, strlen(c));

	qsort(a, n, sizeof(char*), pstrcmp);

//	for(int i=0; i<n; i++)
//		printf("%s", a[i]);

	int maxi = 0;
	int maxLen = -1;
	for(int i=0; i<n-1; i++)
	{
		int matchedLen = comlen(a[i], a[i+1]);
		if(matchedLen > maxLen)
		{
			maxLen = matchedLen;
			maxi = i;
		}
	}
	printf("%.*s\n", maxLen, a[maxi]);
}

