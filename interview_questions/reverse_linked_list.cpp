#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode dummy(0);
        ListNode* p = &dummy;
        while(head != NULL)
        {
            ListNode* next = head->next;
            head->next = p->next;
            p->next = head;
            head = next;
        }
        return dummy.next;
    }

    ListNode* reverseBetween(ListNode* head, int m, int n)
    {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* p = &dummy;

        // find the (m-1)th node
        for(int i=0; i<m-1; ++i)
            p = p->next;

        ListNode* prev = p;
        ListNode* cur = p->next;
        ListNode* tail = cur;
        for(int i=m; i<=n; ++i)
        {
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        p->next = prev;
        tail->next = cur;
        return dummy.next;
    }

    ListNode* rotateRight(ListNode* head, int k) {

        if(k==0 || head == NULL || head->next == NULL)
            return head;

        int n = 0;
        ListNode* p = head;
        ListNode* tail = NULL;
        while(p != NULL)
        {
            ++n;
            tail = p;
            p = p->next;
        }

        k %= n;
        if(k == 0) return head;

        ListNode dummy(0);
        dummy.next = head;
        p = &dummy;

        // find the (n-k)th node
        for(int i=0; i<n-k; ++i)
        {
            p = p->next;
        }

        dummy.next = p->next;
        tail->next = head;
        p->next = NULL;

        return dummy.next;
    }
};


int main()
{
    string input = "[1,2,3,4,5]";
    //string input = "[]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.reverseList(ll);
    printLinkedList(newLL);

    newLL = ss.reverseBetween(newLL, 2, 4);
    printLinkedList(newLL);

    newLL = ss.rotateRight(newLL, 2);
    printLinkedList(newLL);

    destroyLinkedList(newLL);

    return 0;
}
