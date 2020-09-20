#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode 2, 445*/

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
    ListNode* addTwoNumbersII(ListNode* l1, ListNode* l2);
};

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2) {
/*
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in reverse order and each of their nodes contain a single digit. 
    Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Example:
        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 0 -> 8
        Explanation: 342 + 465 = 807.
*/
    int carry = 0;
    ListNode dummy(0);
    ListNode* p = &dummy;
    while(l1 != nullptr || l2 != nullptr || carry != 0) {
        int val = carry;
        if (l1 != nullptr) {
            val += l1->val;
            l1 = l1->next;
        }
        if (l2 != nullptr) {
            val += l2->val;
            l2 = l2->next;
        }

        if(val > 9) {
            carry = 1;
            val -= 10;
        } else {
            carry = 0;
        }

        ListNode* t = new ListNode(val);
        p->next = t; p = t;
    }
    return dummy.next;
}

ListNode* Solution::addTwoNumbersII(ListNode* l1, ListNode* l2) {
/*
    You are given two non-empty linked lists representing two non-negative integers. 
    The most significant digit comes first and each of their nodes contain a single digit. 
    Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Example:
        Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 8 -> 0 -> 7
*/
    stack<int> s1, s2;
    while(l1 != nullptr || l2 != nullptr) {
        if (l1 != nullptr) {
            s1.push(l1->val); l1 = l1->next;
        }
        if (l2 != nullptr) {
            s2.push(l2->val); l2 = l2->next;
        }
    }

    int carry = 0;
    ListNode dummy(0);
    ListNode* p = &dummy;
    while(!s1.empty() || !s2.empty() || carry != 0) {
        int val = carry;
        if(!s1.empty()) {
            val += s1.top();
            s1.pop();
        }

        if(!s2.empty()) {
            val += s2.top();
            s2.pop();
        }

        carry = 0;
        if(val > 9) {
            val -= 10;
            carry = 1;
        }

        ListNode* t = new ListNode(val);
        t->next = p->next; p->next = t;
    }
    return dummy.next;
}

void addTwoNumbers_scaffold(string input1, string input2, string input3) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* l3 = stringToListNode(input3);
    Solution ss;
    ListNode* ans = ss.addTwoNumbers(l1, l2);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") failed.";
    }
}

void addTwoNumbersII_scaffold(string input1, string input2, string input3) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* l3 = stringToListNode(input3);
    Solution ss;
    ListNode* ans = ss.addTwoNumbersII(l1, l2);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") failed.";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running addTwoNumbers tests:";
    TIMER_START(addTwoNumbers);
    addTwoNumbers_scaffold("[3,4,5]", "[7,0,8]", "[0,5,3,1]");
    addTwoNumbers_scaffold("[2,4,3]", "[5,6,4]", "[7,0,8]");
    TIMER_STOP(addTwoNumbers);
    util::Log(logESSENTIAL) << "Running addTwoNumbers tests uses " << TIMER_MSEC(addTwoNumbers) << "ms.";

    util::Log(logESSENTIAL) << "Running addTwoNumbersII tests:";
    TIMER_START(addTwoNumbersII);
    addTwoNumbersII_scaffold("[3,4,5]", "[7,0,8]", "[1,0,5,3]");
    addTwoNumbersII_scaffold("[2,4,3]", "[5,6,4]", "[8,0,7]");
    addTwoNumbersII_scaffold("[7,2,4,3]", "[5,6,4]", "[7,8,0,7]");
    TIMER_STOP(addTwoNumbersII);
    util::Log(logESSENTIAL) << "Running addTwoNumbersII tests uses " << TIMER_MSEC(addTwoNumbersII) << "ms.";

}
