#include "leetcode.h"

using namespace std;

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;

        ListNode* fast = dummy;
        ListNode* slow = dummy;
        while(fast->next != NULL)
        {
            if(n <= 0)
                slow = slow->next;

            fast = fast->next;
            n--;
        }

        if(slow->next != NULL)
            slow->next = slow->next->next;

        head = dummy->next;
        delete dummy;
        return head;
    }
};

int main()
{
    string input = "[1,2,3,4]";
    ListNode* oldHead = stringToListNode(input);
    printLinkedList(oldHead);

    Solution ss;
    ListNode* newHead = ss.removeNthFromEnd(oldHead, 5);
    printLinkedList(newHead);
    destroyLinkedList(newHead);

    return 0;
}
