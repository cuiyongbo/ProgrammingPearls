#include <stdio.h>
#include <stdlib.h>
#include "single_linked_list.h"

void initLinkedList(LinkedList l, int nodeCount)
{
	int i = 0;
	for(; i<nodeCount; i++)
	{ // push_front
		Node* p = allocNodeWithVal(i);
		p->next = l->next;
		l->next = p;
	}
} 

void LinkedList_insert_before_hint(LinkedList l, Node* p, Node* hint)
{
	if(l == NULL)
		return;

	Node* q = l;
	for(; q->next != NULL && q->next != hint;)
		q = q->next;

	p->next = q->next;
	q->next = p;		
}

LinkedList local_merge(LinkedList l, LinkedList r);

int main()
{
	LinkedList l = createLinkedList(); 	

	initLinkedList(l, 5);	
	traverseLinkedList(l);

	Node* p = NULL;
	p = allocNodeWithVal(35);
	LinkedList_insert(l, p, 4);
	p = allocNodeWithVal(35);
	LinkedList_insert(l, p, 0);
	p = allocNodeWithVal(35);
	LinkedList_insert(l, p, 1);
	p = allocNodeWithVal(35);
	LinkedList_insert(l, p, 40);
	traverseLinkedList(l);

	LinkedList_delete(l, 0);
	LinkedList_delete(l, 1);
	LinkedList_delete(l, 78);
	LinkedList_delete(l, 4);
	traverseLinkedList(l);
	
	LinkedList r = createLinkedList();
	initLinkedList(r, 10);
	//l = local_merge(l,r);
	//traverseLinkedList(l);

	LinkedList merged = merge(l,r);
	traverseLinkedList(merged);
	destroyLinkedList(merged);

	destroyLinkedList(r);
	destroyLinkedList(l);
	return 0;
}


// so many bugs 
// if l.merge(r) would be better
LinkedList local_merge(LinkedList l, LinkedList r)
{
	if(r == NULL)
		return l;

	LinkedList result = l;
	Node* lp = l->next;
	Node* rp = r->next;
	r->next = NULL;
	while(lp != NULL && rp->next != NULL)
	{
		if(lp->data > rp->data)
		{
			lp = lp->next;
		}
		else
		{
			Node* p = rp;
			rp = rp->next;
			LinkedList_insert_before_hint(result, p, lp);
		}
	}

	if(rp->next != NULL)
	{
		// append rp to result
		lp = result;
		while (lp->next != NULL)
			lp = lp->next;
		lp->next = rp;
	}
	
	return result;
}

