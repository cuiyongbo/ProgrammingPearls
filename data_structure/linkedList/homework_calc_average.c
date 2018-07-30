#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Node
{
    float data;
    struct Node* next;
} Node; 

typedef Node* LinkedList;

Node* allocNodeWithVal(float val)
{
    Node* n = (Node*)malloc(sizeof(Node));
    n->data = val;
    n->next = NULL;
    return n;
}

void freeNode(Node* p) { free(p); }

LinkedList createLinkedList() 
{
    LinkedList l = (LinkedList)malloc(sizeof(Node));
    l->next = NULL;
    return l;
}

void destroyLinkedList(LinkedList l)
{
    if(l == NULL)
        return;
    
    Node *p, *q; 
    p = l->next;
    while(p != NULL)
    {   
        q = p->next;
        freeNode(p);
        p = q;
    }   
    free(l);
}

void LinkedList_insert(LinkedList l, Node* p)
{
    if(l == NULL)
        return;

    p->next = l->next;
    l->next = p;
}

int getSumAndCount(LinkedList l, float* sumOut)
{
	int count = 0;
    if(l == NULL)
        return count;

	float sum = .0f;
    Node *p = l->next;
    while(p != NULL)
    {
		sum += p->data;
		count++;
        p = p->next;
    }
	*sumOut = sum;
	return count;
}

void calcAverage()
{
	LinkedList l = createLinkedList();
	float x = 0.0f;
	while(1)
	{
		printf("Enter a number, enter 0 to exit: ");
		scanf("%f", &x);
		if(fabs(x) < 1e-5)
			break;
		
		Node* p = allocNodeWithVal(x);
		LinkedList_insert(l, p);
	}
	float sum;
	int count = getSumAndCount(l, &sum); 		

	if(count != 0)
		printf("You have input %d non-zero real number, The average is %f\n", count, sum/count);

	destroyLinkedList(l);
}

int main()
{
	calcAverage();
	return 0;
}
