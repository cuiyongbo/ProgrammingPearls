#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0);
        ListNode* p = &dummy;
        ListNode* prev = NULL;
        ListNode* cur = head;
        stack<ListNode*> s;
        while(cur != NULL)
        {
            prev = cur;
            while(cur != NULL && s.size() < k)
            {
                s.push(cur);
                cur = cur->next;
            }

            if(s.size() == k)
            {
                while(!s.empty())
                {
                    auto n = s.top(); s.pop();
                    p->next = n;
                    p = n;
                    p->next = NULL;
                }
            }
            else
            {
                p->next = prev;
            }
        }
        return dummy.next;
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
    ListNode* newLL = ss.reverseKGroup(ll, 2);
    printLinkedList(newLL);

    destroyLinkedList(newLL);
    return 0;
}
