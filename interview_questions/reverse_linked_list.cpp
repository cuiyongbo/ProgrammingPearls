#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        ListNode* newHead = dummy;
        while(head != NULL)
        {
            ListNode* p = head->next;
            head->next = newHead->next;
            newHead->next = head;
            head = p;
        }
        newHead = dummy->next;
        delete dummy;
        return newHead;
    }
};

int main()
{
    //string input = "[1,2,3,4,5]";
    string input = "[]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.reverseList(ll);
    printLinkedList(newLL);

    destroyLinkedList(newLL);

    return 0;
}
