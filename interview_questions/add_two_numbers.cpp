#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int carray = 0;
        ListNode dummy(0);
        ListNode* p = &dummy;
        while(l1 != NULL && l2 != NULL)
        {
            int val = l1->val + l2->val + carray;
            if(val > 9)
            {
                carray = 1;
                val -= 10;
            }
            else
            {
                carray = 0;
            }

            ListNode* t = new ListNode(val);
            p->next = t;
            p = t;
            l1 = l1->next;
            l2 = l2->next;
        }

        ListNode* remains = (l1 != NULL) ? l1 : l2;
        while(remains != NULL)
        {
            int val = remains->val + carray;
            if(val > 9)
            {
                carray = 1;
                val -= 10;
            }
            else
            {
                carray = 0;
            }

            ListNode* t = new ListNode(val);
            p->next = t;
            p = t;
            remains = remains->next;
        }

        if(carray == 1)
        {
            p->next = new ListNode(1);
        }
        return dummy.next;
    }

    ListNode* addTwoNumbersII(ListNode* l1, ListNode* l2) {
        stack<int> s1, s2;
        while(l1 != NULL)
        {
            s1.push(l1->val);
            l1 = l1->next;
        }

        while(l2 != NULL)
        {
            s2.push(l2->val);
            l2 = l2->next;
        }

        int carray = 0;
        ListNode dummy(0);
        ListNode* p = &dummy;
        while(!s1.empty() || !s2.empty() || carray != 0)
        {
            int val = carray;
            if(!s1.empty())
            {
                val += s1.top();
                s1.pop();
            }

            if(!s2.empty())
            {
                val += s2.top();
                s2.pop();
            }

            carray = 0;
            if(val > 9)
            {
                val -= 10;
                carray = 1;
            }

            ListNode* t = new ListNode(val);
            t->next = p->next;
            p->next = t;
        }
        return dummy.next;
    }
};
