#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {

        if(l1 == NULL)
            return l2;
        else if(l2 == NULL)
            return l1;

        ListNode dummy(0);
        ListNode* p = &dummy;
        while(l1 != NULL && l2 != NULL)
        {
            if(l1->val < l2->val)
            {
                p->next = l1;
                p = l1;
                l1 = l1->next;
            }
            else
            {
                p->next = l2;
                p = l2;
                l2 = l2->next;
            }
        }
        p->next = (l1 != NULL) ? l1 : l2;
        return dummy.next;
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return helper(lists, 0, lists.size());
    }

 private:
    ListNode* helper(vector<ListNode*>& lists, int start, int end)
    {
        if(start >= end)
            return NULL;
        else if(start + 1 == end)
            return lists[start];

        int mid = start + (end - start)/2;
        ListNode* left = helper(lists, start, mid);
        ListNode* right = helper(lists, mid, end);
        return mergeTwoLists(left, right);
    }
};

int main()
{
    vector<ListNode*> lists;
    //vector<string> inputs = {"[1,4,5]", "[1,3,4]", "[2,6]"};
    vector<string> inputs = {"[1,4,5]", "[1,3,4]"};
    for(auto& s : inputs)
    {
        ListNode* ll = stringToListNode(s);
        printLinkedList(ll);
        lists.push_back(ll);
    }

    Solution ss;
    ListNode* newLL = ss.mergeKLists(lists);
    printLinkedList(newLL);
    destroyLinkedList(newLL);

    return 0;
}
