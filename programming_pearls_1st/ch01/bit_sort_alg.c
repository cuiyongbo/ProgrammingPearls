/*
	This version operates on int[] instead of char[]
*/

#include <stdio.h>

#define element_of(a) (sizeof(a)/sizeof(a[0]))

#define SHIFT 5
#define MASK 0x1F
#define BITS_PER_WORD 32	/* 32 = 2^5 */

#define NUM	10000000
int a[1 + NUM/BITS_PER_WORD];

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
