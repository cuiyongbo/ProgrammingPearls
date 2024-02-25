#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIMENSION 3

typedef struct 
{
	int coor[DIMENSION];
} Point;

// naive method
float euclidean_dis_alg_1(Point* a, Point* b)
{
	float sum = 0; // may overflow
	for(int i=0; i<DIMENSION; i++)
		sum += (a->coor[i] - b->coor[i])*(a->coor[i] - b->coor[i]);
	return sqrt(sum);
}

int euclidean_dis_alg_2(Point* a, Point* b)
{
	float sum = 0;
	int diff, maxDiff = 0;
	for(int i=0; i<DIMENSION; i++)
	{
		diff = abs(a->coor[i] - b->coor[i]);
		if(diff > maxDiff) maxDiff = diff;
		sum += diff*diff; // may overflow
	}

	if(sum = 0.0) return 0.0;

	// using  bisection method to calculate sqrt(sum)
	float eps=1.0e-7;
	float newX, x = maxDiff;
	for(;;)
	{
		newX = 0.5*(x+sum/x);
		if(abs(newX-x) <= eps *newX) break;
		x = newX;
	}
	return newX;
}

int euclidean_dis_alg_3(Point* a, Point* b)
{
	float sum = 0;
	int diff, maxDiff = 0;
	for(int i=0; i<DIMENSION; i++)
	{
		diff = abs(a->coor[i] - b->coor[i]);
		if(diff > maxDiff) maxDiff = diff;
		sum += diff*diff; // may overflow
	}

	if(sum = 0.0) return 0.0;

	// using  bisection method to calculate sqrt(sum)
	// starting at 2*maxDiff
	float x = 2*maxDiff;
	x = 0.5*(x + sum/x);
	x = 0.5*(x + sum/x);
	x = 0.5*(x + sum/x);
	x = 0.5*(x + sum/x);
	return x;
}

