#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* mergeSort(ListNode* head);
    ListNode* insertionSortList(ListNode* head);
    void reorderList(ListNode* head);

private:
    ListNode* partition(ListNode* head)
    {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* slow = &dummy;
        ListNode* fast = head;
        while(fast != NULL && fast->next != NULL)
        {
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode* tmp = slow->next;
        slow->next = NULL;
        return tmp;
    }

    ListNode* merge(ListNode* left, ListNode* right)
    {
        if(left == NULL)
            return right;
        else if(right == NULL)
            return left;

        ListNode dummy(0);
        ListNode* p = &dummy;
        while(left != NULL && right != NULL)
        {
            if(left->val < right->val)
            {
                p->next = left;
                p = left;
                left = left->next;
            }
            else
            {
                p->next = right;
                p = right;
                right = right->next;
            }
        }
        p->next = (left != NULL) ? left : right;
        return dummy.next;
    }
};

ListNode* Solution::mergeSort(ListNode* head)
{
    if(head == NULL || head->next == NULL)
        return head;

    ListNode* mid = partition(head);
    ListNode* left = mergeSort(head);
    ListNode* right = mergeSort(mid);
    return merge(left, right);
}

ListNode* Solution::insertionSortList(ListNode* head)
{
    ListNode dummy(0);
    while(head != NULL)
    {
        int val = head->val;
        ListNode* p = &dummy;
        while(p->next != NULL && p->next->val < val)
        {
            p = p->next;
        }

        ListNode* next1 = p->next;
        ListNode* next2 = head->next;

        p->next = head;
        head->next = next1;

        head = next2;
    }
    return dummy.next;
}

void Solution::reorderList(ListNode* head)
{
    if(head == NULL || head->next == NULL)
        return;

    ListNode* slow = head;
    ListNode* fast = head;
    while(fast != NULL && fast->next != NULL)
    {
        fast = fast->next->next;
        slow = slow->next;
    }

    // split list into left and right sublist
    // leftNodeCount >= rightNodeCount
    ListNode* right = slow->next;
    slow->next = NULL;

    stack<ListNode*> s;
    while(right != NULL)
    {
        s.push(right);
        right = right->next;
    }

    ListNode* left = head;
    while(!s.empty())
    {
        auto node = s.top(); s.pop();

        ListNode* next = left->next;
        left->next = node;
        node->next = next;
        left = next;
    }
}

int main()
{
    // string input = "[]";
    //string input = "[1,2,3,4,5]";
    string input = "[9,2,2,7,3,4,5]";
    //string input = "[1,2,2,3,3]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.mergeSort(ll);
    printLinkedList(newLL);

    destroyLinkedList(newLL);
    return 0;
}
