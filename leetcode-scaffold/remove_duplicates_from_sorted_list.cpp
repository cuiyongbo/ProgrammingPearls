#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* deleteDuplicatesII(ListNode* head)
    {
        ListNode* dummy = new ListNode(0);
        ListNode* newHead = dummy;
        while(head != NULL)
        {
            if(head->next == NULL)
            {
                newHead->next = head;
                newHead = head;
                head = head->next;
            }
            else
            {
                if(head->val == head->next->val)
                {
                    int val = head->val;
                    while(head != NULL && head->val == val)
                    {
                        head = head->next;
                    }
                }
                else
                {
                    newHead->next = head;
                    newHead = head;
                    head = head->next;
                }
            }
        }
        newHead->next = NULL;
        newHead = dummy->next;
        delete dummy;
        return newHead;
    }

    ListNode* deleteDuplicates(ListNode* head)
    {

        if(head == NULL || head->next == NULL)
            return head;

        ListNode* p = head;
        ListNode* tmp = head;
        while(p != NULL)
        {
            if(tmp->val != p->val)
            {
                tmp->next = p;
                tmp = p;
            }
            p = p->next;
        }
        tmp->next = NULL;

        return head;
    }
};

int main()
{
    // string input = "[]";
    //string input = "[1,2,3,4,5]";
    //string input = "[1,2,2,3,3,4,5]";
    string input = "[1,2,2,3,3]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.deleteDuplicates(ll);
    printLinkedList(newLL);

    destroyLinkedList(newLL);
    return 0;
}

