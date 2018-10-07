#include "ppearls.h"

int ssearch_1(int* a, int length, int key)
{
	for(int i=0; i<length; i++)
	{
		if(a[i] == key)
			return i;
	}
	return -1;
}

int ssearch_2(int* a, int length, int key)
{
	int sentinel = a[length];
	a[length] = key;
	int p;
	for(p=0;;p++)
	{
		if(a[p] == key)
			break;	
	}
	a[length] = sentinel;
	return (p<length) ? p : -1;
}

int ssearch_3(int* a, int length, int key)
{
	int sentinel = a[length];
	a[length] = key;
	int p;	
	for(p=0;;p<<=3)
	{
		if(a[i] == key) break;
		if(a[i+1] == key) {i+=1; break;}
		if(a[i+2] == key) {i+=2; break;}
		if(a[i+3] == key) {i+=3; break;}
		if(a[i+4] == key) {i+=4; break;}
		if(a[i+5] == key) {i+=5; break;}
		if(a[i+6] == key) {i+=6; break;}
		if(a[i+7] == key) {i+=7; break;}
	}
	a[length] = sentinel;
	return (p<length) ? p : -1;
}

