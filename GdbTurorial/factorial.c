# include <stdio.h>

int main()
{
	int num;
	printf ("Enter the number: ");
	scanf ("%d", &num );

	int j = 1;
	for (int i=1; i<=num; ++i)
		j=j*i;    

	printf(" %d! = %d\n",num,j);
}

