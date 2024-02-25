#include <iostream>

#include "leetcode.h"

using namespace osrm;

/*
    Leetcode 707
    Design your implementation of the linked list.
    You can choose to use the singly linked list or the doubly linked list.
    A node in a singly linked list should have two attributes: `val` and `next`.
    val is the value of the current node, and `next` is a pointer/reference to the next node.
    If you want to use the doubly linked list, you will need one more attribute `prev` to
    indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

    Implement these functions in your linked list class:

    `get(index)` : Get the value of the index-th node in the linked list. If the index is invalid, return -1.
    `addAtHead(val)` : Add a node of value val before the first element of the linked list.
    `addAtTail(val)` : Append a node of value val to the last element of the linked list.
    `addAtIndex(index, val)` : Add a node of value val before the index-th node in the linked list.
            If index equals to the length of linked list, the node will be appended to the end of linked list.
            If index is greater than the length, or index is negative,  the node will not be inserted.
    `deleteAtIndex(index)` : Delete the index-th node in the linked list, if the index is valid.
*/

class MyLinkedList {
public:
    MyLinkedList();
    ~MyLinkedList();

    int get(int index);
    void addAtHead(int val);
    void addAtTail(int val);
    void addAtIndex(int index, int val);
    void deleteAtIndex(int index);

    ListNode* data() const { return m_dummyNode.next; }
    int nodeCount() const { return m_nodeCount; }

    MyLinkedList(const MyLinkedList&) = delete;
    MyLinkedList& operator=(const MyLinkedList&) = delete;

private:
    int m_nodeCount;
    ListNode m_dummyNode;
};

MyLinkedList::MyLinkedList()
    :m_nodeCount(0) {
}

MyLinkedList::~MyLinkedList() {
    destroyLinkedList(m_dummyNode.next);
}

int MyLinkedList::get(int index) {
    if (index >= m_nodeCount || index < 0) {
        return -1;
    }

    ListNode* p = m_dummyNode.next;
    for (int curIndex = 0; curIndex != index; ++curIndex) {
        p = p->next;
    }
    return p->val;
}

void MyLinkedList::addAtHead(int val) {
    addAtIndex(0, val);
}

void MyLinkedList::addAtTail(int val) {
    addAtIndex(m_nodeCount, val);
}

void MyLinkedList::addAtIndex(int index, int val) {
    if (index > m_nodeCount || index < 0) {
        return;
    }

    ListNode* p = &m_dummyNode;
    ListNode* q = m_dummyNode.next;
    for (int curIndex = 0; q != nullptr && curIndex != index; ++curIndex) {
        p = q;
        q = q->next;
    }

    ListNode* t = new ListNode(val);
    t->next = q;
    p->next = t;

    ++m_nodeCount;
}

void MyLinkedList::deleteAtIndex(int index) {
    if (index >= m_nodeCount || index < 0) {
        return;
    }

    ListNode* p = &m_dummyNode;
    ListNode* q = m_dummyNode.next; // the node to delete
    for (int curIndex = 0; q != nullptr && curIndex != index; ++curIndex) {
        p = q;
        q = q->next;
    }

    p->next = q->next;
    q->next = nullptr;

    delete q;

    --m_nodeCount;
}

void basic_test() {
    MyLinkedList* myList = new MyLinkedList;

    myList->addAtHead(1);   // 1
    myList->addAtTail(3);   // 1,3
    myList->addAtIndex(1, 2);  // 1,2,3

    ListNode* expected = stringToListNode("[1,2,3]");
    ListNode* actual = myList->data();
    if (!list_equal(actual, expected)) {
        util::Log(logERROR) << "MyLinkedList insertion failed";
        return;
    }

    int val = myList->get(0);
    if (val != 1) {
        util::Log(logERROR) << "MyLinkedList::get failed, expect: " << 1 << ", acutal: " << val;
        return;
    }

    val = myList->get(2);
    if (val != 3) {
        util::Log(logERROR) << "MyLinkedList::get failed, expect: " << 3 << ", acutal: " << val;
        return;
    }

    val = myList->get(4);
    if (val != -1) {
        util::Log(logERROR) << "MyLinkedList::get failed, expect: " << -1 << ", acutal: " << val;
        return;
    }

    myList->deleteAtIndex(1);
    val = myList->get(1);
    if (val != 3) {
        util::Log(logERROR) << "MyLinkedList::deleteAtIndex failed, expect: " << 3 << ", acutal: " << val;
        return;
    }

    delete myList;
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();
    basic_test();
}
