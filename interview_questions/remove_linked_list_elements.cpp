#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        ListNode* dummy = new ListNode(0);
        ListNode* newHead = dummy;
        while(head != NULL)
        {
            if(head->val != val)
            {
                newHead->next = head;
                newHead = head;
            }
            head = head->next;
        }
        newHead->next = NULL;
        newHead = dummy->next;
        delete dummy;
        return newHead;
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
    // ListNode* newLL = ss.removeElements(ll, 2);
    ListNode* newLL = ss.removeElements(ll, 3);
    printLinkedList(newLL);

    destroyLinkedList(newLL);
    return 0;
}

