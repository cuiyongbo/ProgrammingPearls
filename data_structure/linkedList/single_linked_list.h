#pragma once

typedef struct Node
{
	int data;
	struct Node* next;
} Node; 

typedef Node* LinkedList;

Node* allocNodeWithVal(int val);
void freeNode(Node* p);

LinkedList createLinkedList(); 
void destroyLinkedList(LinkedList l);

void traverseLinkedList(LinkedList l);

// Add Node as the ith node in the linked list. 
// hint < 1 --> push_front
// hint > linkSize --> push_back
void LinkedList_insert(LinkedList l, Node* p, int hint);

// Remove Node as the ith node in the linked list. 
// hint < 1 --> pop_front
// hint > linkSize --> pop_back
void LinkedList_delete(LinkedList l, int hint);

LinkedList merge(LinkedList l, LinkedList r);

void LinkedList_insertionSort(LinkedList l);
