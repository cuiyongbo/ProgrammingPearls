/*
	This version operates on char[]
*/

#include <stdio.h>
#include <limits.h> // CHAR_BIT

#define element_of(a) (sizeof(a)/sizeof(a[0]))

#define SHIFT 3
#define MASK 0x07

#define NUM	10000000
char a[(NUM+7)/CHAR_BIT]; // make sure that element_of(a) * CHAR_BIT >= NUM

void set(int i) {        a[i>>SHIFT] |=  (1<<(i & MASK)); }
void clr(int i) {        a[i>>SHIFT] &= ~(1<<(i & MASK)); }
int  test(int i){ return !!(a[i>>SHIFT] & (1<<(i & MASK))); }

int main()
{
	int i;
	int n = element_of(a);
	for(i=0; i<n; i++) a[i] = 0;

	while (scanf("%d", &i) != EOF && i<NUM)
        set(i);

	for (i = 0; i < NUM; i++)
	{
        if (test(i)) printf("%d\n", i);
	}
	return 0;
}
