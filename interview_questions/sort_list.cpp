#include "leetcode.h"

using namespace std;

class Solution {
public:
    ListNode* sortList(ListNode* head)
    {
        if(head == NULL || head->next == NULL)
            return head;

        ListNode* mid = partition(head);
        ListNode* left = sortList(head);
        ListNode* right = sortList(mid);
        return merge(left, right);
    }

private:
    ListNode* partition(ListNode* head)
    {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* slow = &dummy;
        ListNode* fast = head;
        while(fast != NULL && fast->next != NULL)
        {
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode* tmp = slow->next;
        slow->next = NULL;
        return tmp;
    }

    ListNode* merge(ListNode* left, ListNode* right)
    {
        if(left == NULL)
            return right;
        else if(right == NULL)
            return left;

        ListNode dummy(0);
        ListNode* p = &dummy;
        while(left != NULL && right != NULL)
        {
            if(left->val < right->val)
            {
                p->next = left;
                p = left;
                left = left->next;
            }
            else
            {
                p->next = right;
                p = right;
                right = right->next;
            }
        }
        p->next = (left != NULL) ? left : right;
        return dummy.next;
    }
};

int main()
{
    // string input = "[]";
    //string input = "[1,2,3,4,5]";
    string input = "[9,2,2,7,3,4,5]";
    //string input = "[1,2,2,3,3]";
    ListNode* ll = stringToListNode(input);
    printLinkedList(ll);

    Solution ss;
    ListNode* newLL = ss.sortList(ll);
    printLinkedList(newLL);

    destroyLinkedList(newLL);
    return 0;
}
