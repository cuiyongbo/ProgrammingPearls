#include <stdio.h>
#include <stdlib.h>
#include "single_linked_list.h"

Node* allocNodeWithVal(int val)
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

void traverseLinkedList(LinkedList l)
{
	if(l == NULL)
		return;
	
	Node *p = l->next;
	while(p != NULL)
	{
		printf("%d --> ", p->data);
		p = p->next;
	}
	printf("NIL\n");
}

// Add Node as the ith node in the linked list. 
// hint < 1 --> push_front
// hint > linkSize --> push_back
void LinkedList_insert(LinkedList l, Node* p, int hint)
{
	if(l == NULL)
		return;

	Node* q = l;
	int count = 1;
	for(; q->next != NULL && count < hint; count++)
		q = q->next;

	p->next = q->next;
	q->next = p;		
}

// Remove Node as the ith node in the linked list. 
// hint < 1 --> pop_front
// hint > linkSize --> pop_back
void LinkedList_delete(LinkedList l, int hint)
{
	if(l == NULL)
		return;

	Node* q = l;
	int count = 1;
	for(; q->next->next != NULL && count < hint; count++)
	//for(; q->next != NULL && count < hint; count++) // segment fault 11
		q = q->next;

	Node* p = q->next;
	q->next = p->next;
	freeNode(p);		
}

LinkedList merge(LinkedList l, LinkedList r)
{
	if(l == NULL && r != NULL)
		return r;
	if(l != NULL && r == NULL)
		return l;

	LinkedList result = createLinkedList();
	Node* lp = l->next;
	Node* rp = r->next;
	l->next = NULL;
	r->next = NULL;
	Node* p = NULL;
	while(lp != NULL && rp != NULL)
	{
		if(lp->data > rp->data)
		{
			p = lp;
			lp = lp->next;
		}
		else
		{
			p = rp;
			rp = rp->next;
		}
		LinkedList_insert(result, p, 0);
	}

	while(lp != NULL)
	{
		p = lp;
		lp = lp->next;
		LinkedList_insert(result, p, 0);
	}

	while(rp != NULL)
	{
		p = rp;
		rp = rp->next;
		LinkedList_insert(result, p, 0);
	}
	
	return result;
}

void LinkedList_insertionSort(LinkedList l)
{
	// zero or one element
	if(l->next == NULL || l->next->next == NULL)
		return;

	LinkedList newHead = l;

	Node* p = l->next;
	while(p->next )	



}
