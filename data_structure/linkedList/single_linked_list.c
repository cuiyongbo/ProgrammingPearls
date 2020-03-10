#include <stdio.h>
#include <stdlib.h>

typedef struct Node
{
	int data;
	struct Node* next;
} Node; 

typedef Node* LinkedList;

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

void initLinkedList(LinkedList l, int nodeCount)
{
	// const int nodeCount = 10;		
	int i = 0;
	for(; i<nodeCount; i++)
	{ // push_front
		//Node* p = (Node*)malloc(sizeof(Node));
		//p->data = i;
		Node* p = allocNodeWithVal(i);
		p->next = l->next;
		l->next = p;
	}
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
	
	destroyLinkedList(l);

	return 0;
}


