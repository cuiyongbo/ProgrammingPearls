#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 2, 445, 206, 24*/

class Solution
{
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
    ListNode* addTwoNumbers_445(ListNode* l1, ListNode* l2);
    ListNode* reverseList(ListNode* head);
    ListNode* swapPairs(ListNode* head);
};

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2)
{
    /*
        You are given two non-empty linked lists representing two non-negative integers.
        The digits are stored in reverse order and each of their nodes contain a single digit.
        Add the two numbers and return it as a linked list.

        You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    */

    int carry = 0;
    ListNode dummy(0);
    ListNode* head = &dummy;
    while(l1 != NULL && l2 != NULL)
    {
        int s = l1->val + l2->val + carry;
        if(s > 9)
        {
            s -= 10;
            carry = 1;
        }
        else
        {
            carry = 0;
        }

        head->next = new ListNode(s);
        head = head->next;

        l1 = l1->next;
        l2 = l2->next;
    }

    ListNode* l3 = (l1 != NULL) ? l1 : l2;
    while(l3 != NULL)
    {
        int s = l3->val + carry;
        if(s > 9)
        {
            s -= 10;
            carry = 1;
        }
        else
        {
            carry = 0;
        }

        head->next = new ListNode(s);
        head = head->next;

        l3 = l3->next;
    }

    // for last carry
    if(carry != 0)
    {
        head->next = new ListNode(1);
        head = head->next;
    }

    // remove leading zero(s)
    ListNode* r1 = reverseList(dummy.next);
    while(r1 != NULL && r1->val == 0 && r1->next != NULL)
    {
        r1 = r1->next;
    }
    return reverseList(r1);
}

ListNode* Solution::addTwoNumbers_445(ListNode* l1, ListNode* l2)
{
    /*
        Same as last problem excepted that the most significant digit comes first in the list.
    */

    stack<ListNode*> s1, s2;
    while(l1 != NULL || l2 != NULL)
    {
        if (l1 != NULL)
        {
            s1.push(l1);
            l1 = l1->next;
        }

        if (l2 != NULL)
        {
            s2.push(l2);
            l2 = l2->next;
        }
    }

    int carry = 0;
    ListNode dummy(0);
    ListNode* head = &dummy;
    while(!s1.empty() || !s2.empty())
    {
        int s = carry;
        if (!s1.empty())
        {
            s += s1.top()->val;
            s1.pop();
        }

        if (!s2.empty())
        {
            s += s2.top()->val;
            s2.pop();
        }

        if(s > 9)
        {
            s -= 10;
            carry = 1;
        }
        else
        {
            carry = 0;
        }

        ListNode* t = new ListNode(s);
        t->next = head->next;
        head->next = t;
    }

    // for last carry
    if(carry != 0)
    {
        ListNode* t = new ListNode(1);
        t->next = head->next;
        head->next = t;
    }

    // remove leading zero(s)
    ListNode* r1 = dummy.next;
    while(r1 != NULL && r1->val == 0 && r1->next != NULL)
    {
        r1 = r1->next;
    }
    return r1;
}

ListNode* Solution::reverseList(ListNode* head)
{
    ListNode dummy(0);
    ListNode* p = &dummy;
    while(head != NULL)
    {
        ListNode* t = head->next;
        head->next = p->next;
        p->next = head;
        head = t;
    }
    return dummy.next;
}

ListNode* Solution::swapPairs(ListNode* head)
{
    /*
        Given a linked list, swap every two adjacent nodes and return its head.
        You may not modify the values in the list's nodes, only nodes itself may be changed.
    */

    ListNode dummy(0);
    ListNode* p = &dummy;
    stack<ListNode*> s;
    while(head != NULL)
    {
        if(s.size() == 2)
        {
            // push back
            while(!s.empty())
            {
                ListNode* t = new ListNode(s.top()->val);
                p->next = t;
                p = t;
                s.pop();
            }
        }

        s.push(head);
        head = head->next;
    }

    while(!s.empty())
    {
        ListNode* t = new ListNode(s.top()->val);
        p->next = t;
        p = t;
        s.pop();
    }

    return dummy.next;
}

void addTwoNumbers_scaffold(string input1, string input2, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.addTwoNumbers(l1, l2);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(l1);
    destroyLinkedList(l2);
    destroyLinkedList(l3);
}

void addTwoNumbers_445_scaffold(string input1, string input2, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.addTwoNumbers_445(l1, l2);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(l1);
    destroyLinkedList(l2);
    destroyLinkedList(l3);
}

void reverseList_scaffold(string input1, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.reverseList(l1);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(l1);
    destroyLinkedList(l3);
}

void swapPairs_scaffold(string input1, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.swapPairs(l1);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(l1);
    destroyLinkedList(l3);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log() << "Running reverseList tests:";
    reverseList_scaffold("[1,2,3,4,5]", "[5,4,3,2,1]");
    reverseList_scaffold("[1]", "[1]");
    reverseList_scaffold("[]", "[]");

    util::Log() << "Running swapPairs tests:";
    swapPairs_scaffold("[1,2,3,4,5]", "[2,1,4,3,5]");
    swapPairs_scaffold("[1,2,3,4]", "[2,1,4,3]");
    swapPairs_scaffold("[1]", "[1]");
    swapPairs_scaffold("[]", "[]");

    util::Log() << "Running addTwoNumbers tests:";
    addTwoNumbers_scaffold("[2,4,3]", "[6,6,3]", "[8,0,7]");
    addTwoNumbers_scaffold("[1]", "[9]", "[0, 1]");
    addTwoNumbers_scaffold("[1]", "[9,9,9]", "[0,0,0,1]");
    addTwoNumbers_scaffold("[0,0,0]", "[0,0,0]", "[0]");

    util::Log() << "Running addTwoNumbers_445 tests:";
    addTwoNumbers_445_scaffold("[2,4,3]", "[6,6,3]", "[9,0,6]");
    addTwoNumbers_445_scaffold("[1]", "[9]", "[1, 0]");
    addTwoNumbers_445_scaffold("[1]", "[9,9,9]", "[1,0,0,0]");
    addTwoNumbers_445_scaffold("[0,0,0]", "[0,0,0]", "[0]");
}
