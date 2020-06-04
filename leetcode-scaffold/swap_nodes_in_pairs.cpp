#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        ListNode* newHead = dummy;
        stack<ListNode*> s;
        while(head != NULL)
        {
            if(s.size() == 2)
            {
                while(!s.empty())
                {
                    auto node = s.top(); s.pop();
                    newHead->next = node;
                    newHead = node;
                    newHead->next = NULL;
                }
            }
            s.push(head);
            head = head->next;
        }

        while(!s.empty())
        {
            auto node = s.top(); s.pop();
            newHead->next = node;
            newHead = node;
            newHead->next = NULL;
        }

        newHead = dummy->next;
        delete dummy;
        return newHead;
    }
};

int main()
{
    string input = "[1,2,3,4]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.swapPairs(ll);
    printLinkedList(newLL);

    destroyLinkedList(newLL);

    return 0;
}
