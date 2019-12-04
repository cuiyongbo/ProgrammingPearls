#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode leftDummy(0), rightDummy(0);
        ListNode* left = &leftDummy;
        ListNode* right = &rightDummy;
        while(head != NULL)
        {
            if(head->val < x)
            {
                left->next = head;
                left = head;
            }
            else
            {
                right->next = head;
                right = head;
            }
            head = head->next;
        }

        right->next = NULL;
        left->next = rightDummy.next;
        return leftDummy.next;
    }
};
