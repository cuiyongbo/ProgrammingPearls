#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 2, 445, 206, 24, 160, 203, 82, 83, 86, 19, 25 */

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
    ListNode* addTwoNumber_445(ListNode* l1, ListNode* l2);
    ListNode* reverseList(ListNode* head);
    ListNode* swapPairs(ListNode* head);
    ListNode* reverseKGroup(ListNode* head, int k);
    ListNode* getIntersectionNode(ListNode* l1, ListNode* l2);
    ListNode* removeElements(ListNode* head, int val);
    ListNode* deleteDuplicates_083(ListNode* head);
    ListNode* deleteDuplicates_082(ListNode* head);
    ListNode* partition(ListNode* head, int x);
    ListNode* removeNthFromEnd(ListNode* head, int n);
};

ListNode* Solution::reverseList(ListNode* head) {
    ListNode dummy;
    ListNode* p = &dummy;
    while (head != nullptr) {
        ListNode* tmp = head->next; head->next = nullptr;
        head->next = p->next; p->next = head; // push_front
        head = tmp;
    }
    return dummy.next;
}

ListNode* Solution::reverseKGroup(ListNode* head, int k) {
/*
Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
You may not alter the values in the list's nodes, only nodes themselves may be changed.

Examples:
    Input: head = [1,2,3,4,5], k = 2
    Output: [2,1,4,3,5]
    Input: head = [1,2,3,4,5], k = 3
    Output: [3,2,1,4,5]
*/
    ListNode dummy;
    ListNode* p = &dummy;
    std::stack<ListNode*> st;
    while (head != nullptr) {
        if (st.size() == k) {
            while (!st.empty()) {
                auto t = st.top(); st.pop();
                p->next = t; p = p->next; // push_back
            }
        }
        // 别想太多, 这里就得把 header 先给孤立掉
        ListNode* tmp = head->next; head->next = nullptr;
        st.push(head);
        head = tmp;
    }
    if (st.size() == k) {
        while (!st.empty()) {
            auto t = st.top(); st.pop();
            p->next = t; p = p->next; // push_back
        }
    } else { // keep the original order when there are less nodes than k 
        while (!st.empty()) {
            auto t = st.top(); st.pop();
            t->next = p->next; p->next = t; // push_front
        }
    }
    return dummy.next;
}

ListNode* Solution::swapPairs(ListNode* head) {
/*
    Given a linked list, swap every two adjacent nodes and return its head.
    You may not modify the values in the list's nodes, only nodes itself may be changed.
*/
    return reverseKGroup(head, 2);
}

ListNode* Solution::removeNthFromEnd(ListNode* head, int n) {
/*
Given the head of a linked list, remove the nth node from the end of the list and return its head.
Examples:
    Input: head = [1,2,3,4,5], n = 2
    Output: [1,2,3,5]
    Input: head = [1], n = 1
    Output: []
    Input: head = [1,2], n = 1
    Output: [1]
*/
    std::stack<ListNode*> st;
    while (head != nullptr) {
        st.push(head);
        ListNode* tmp = head->next;
        head->next = nullptr; // debug only
        head = tmp;
    }
    ListNode dummy;
    ListNode* p = &dummy;
    int index = 1;
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        if (index++ == n) {
            continue;
        }
        t->next = p->next; p->next = t; // push_front
    }
    return dummy.next;
}

ListNode* Solution::partition(ListNode* head, int x) {
/*
Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.
Examples:
    Input: head = [1,4,3,2,5,2], x = 3
    Output: [1,2,2,4,3,5]
    Input: head = [2,1], x = 2
    Output: [1,2]
*/
    ListNode dummy1, dummy2;
    ListNode* p1 = &dummy1;
    ListNode* p2 = &dummy2;
    while (head != nullptr) {
        ListNode* tmp = head->next; head->next = nullptr;
        if (head->val < x) {
            p1->next = head; p1 = p1->next; // push_back
        } else {
            p2->next = head; p2 = p2->next; // push_back
        }
        head = tmp;
    }
    p1->next = dummy2.next;
    return dummy1.next;
}

ListNode* Solution::deleteDuplicates_082(ListNode* head) {
/*
Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.
Examples:
    Input: head = [1,1,2]
    Output: [2]
    Input: head = [1,1,2,3,3]
    Output: [2]
    Input: head = [1,2,3,3,4,4,5]
    Output: [1,2,5]
*/
    std::stack<std::pair<ListNode*, int>> st;
    while (head != nullptr) {
        ListNode* tmp = head->next; head->next = nullptr;
        if (st.empty()) {
            st.emplace(head, 1);
        } else if (st.top().first->val == head->val) {
            st.top().second++;
        } else {
            st.emplace(head, 1);
        }
        head = tmp;
    }
    ListNode dummy;
    ListNode* p = &dummy;
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        if (t.second == 1) {
            t.first->next = p->next; p->next = t.first; // push_front
        }
    }
    return dummy.next;
}

ListNode* Solution::deleteDuplicates_083(ListNode* head) {
/*
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
Examples:
    Input: head = [1,1,2]
    Output: [1,2]
    Input: head = [1,1,2,3,3]
    Output: [1,2,3]
*/
    std::stack<std::pair<ListNode*, int>> st;
    while (head != nullptr) {
        ListNode* tmp = head->next; head->next = nullptr;
        if (st.empty()) {
            st.emplace(head, 1);
        } else if (st.top().first->val == head->val) {
            st.top().second++;
        } else {
            st.emplace(head, 1);
        }
        head = tmp;
    }
    ListNode dummy;
    ListNode* p = &dummy;
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        t.first->next = p->next; p->next = t.first; // push_front
    }
    return dummy.next;
}

ListNode* Solution::removeElements(ListNode* head, int val) {
/*
Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.
Examples:
    Input: head = [1,2,6,3,4,5,6], val = 6
    Output: [1,2,3,4,5]
    Input: head = [1,2,6,3,4,5,6], val = 1
    Output: [2,6,3,4,5,6]
    Input: head = [], val = 1
    Output: []
    Input: head = [7,7,7,7], val = 7
    Output: []
*/

{
    ListNode dummy;
    ListNode* p = &dummy;
    while (head != nullptr) {
        if (head->val == val) {
            head = head->next;
            continue;
        }
        ListNode* tmp = head->next; head->next = nullptr;
        p->next = head; p = p->next; // push_back
        head = tmp;
    }
    return dummy.next;
}

    ListNode dummy;
    ListNode* p = &dummy;
    while (head != nullptr) {
        if (head->val == val) {
            head = head->next;
            continue;
        }
        ListNode* tmp = head->next; head->next = nullptr;
        p->next = head; p = p->next;
        head = tmp;
    }
    return dummy.next;
}

ListNode* Solution::addTwoNumbers(ListNode* l1, ListNode* l2) {
/*
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in LSB-first order and each of their nodes contain a single digit. 
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
    while (l1 != nullptr || l2 != nullptr || carry != 0) {
        ListNode* node = new ListNode(0);
        node->val += carry; carry = 0;
        if (l1 != nullptr) {
            node->val += l1->val;
            l1 = l1->next;
        }
        if (l2 != nullptr) {
            node->val += l2->val;
            l2 = l2->next;
        }
        if (node->val > 9) {
            carry = 1;
            node->val -= 10;
        }
        p->next = node; p = p->next; // push_back
    }
    return dummy.next;
}

ListNode* Solution::addTwoNumber_445(ListNode* l1, ListNode* l2) {
/*
    You are given two non-empty linked lists representing two non-negative integers. 
    The most significant digit comes first and each of their nodes contain a single digit. 
    Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Example:
        Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 8 -> 0 -> 7
*/
    std::stack<int> s1, s2;
    while (l1 != nullptr || l2 != nullptr) {
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
    while (!s1.empty() || !s2.empty() || carry != 0) {
        ListNode* node = new ListNode;
        node->val += carry; carry = 0;
        if (!s1.empty()) {
            node->val += s1.top(); s1.pop();
        }
        if (!s2.empty()) {
            node->val += s2.top(); s2.pop();
        }
        if (node->val > 9) {
            node->val -= 10;
            carry = 1;
        }
        node->next = p->next; p->next = node; // push_front
    }
    return dummy.next;
}

ListNode* Solution::getIntersectionNode(ListNode* l1, ListNode* l2) {
/*
    Write a program to find the node at which the intersection of two singly linked lists begins.
    Hint: 
        suppose we concatenate two lists together respectively, then we get two virtual lists as followings:
            [3][2,3]
            [2,3][3]
            loop 1: 3,2 --> not equal --> null, 3 
            loop 2: null, 3 --> not equal --> 2, null
            loop 3: 2, null --> not equal --> 3, 3   
            loop 4: 3, 3 -> equal,  ok, we got the intersected node
        then traverse these two virtual lists step by step, and we will coincide at the intersected node

    Examples:

    Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
    Output: Reference of the node with value = 8
    Input Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect). From the head of A,
    it reads as [4,1,8,4,5]. From the head of B, it reads as [5,0,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.

             4+------> 1+---+
                            |
                            v
                            8+------->4+----->5
                            ^
                            |
    5+------->0+-------->1+--

    1: 4, 5 -> x -> 1, 0
    2: 1, 0 -> x -> 8, 1
    3: 8, 1 -> x -> 4, 8
    4: 4, 8 -> x -> 5, 4
    5: 5, 4 -> x -> null, 5
    6: null, 5 -> x -> 5, null
    6: 5, null -> x -> 0, 4
    7: 0, 4 -> x -> 1, 1
    8: 1, 1 -> x -> 8, 8
    9: 8, 8 -> o -> return

    Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
    Output: Intersected at '2'

    Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
    Output: null

    Notes:

    If the two linked lists have no intersection at all, return null.
    The linked lists must retain their original structure after the function returns.
    You may assume there are no cycles anywhere in the entire linked structure.
    Your code should preferably run in O(n) time and use only O(1) memory.
*/

    // 1. iterate over one list, and save nodes into a set, test node existence against the set when iterating the other list, failed to meet the requirement of the running time and space complexity
    ListNode* p1 = l1;
    ListNode* p2 = l2;
    while (p1 != p2) {
        p1 = p1==nullptr ? l2 : p1->next;
        p2 = p2==nullptr ? l1 : p2->next;
    }
    return p1;
}


void addTwoNumbers_scaffold(std::string input1, std::string input2, std::string input3, int func_no) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* l3 = stringToListNode(input3);
    Solution ss;
    ListNode* ans = nullptr;
    if (func_no == 2) {
        ans = ss.addTwoNumbers(l1, l2);
    } else if (func_no == 445) {
        ans = ss.addTwoNumber_445(l1, l2);
    } else {
        util::Log(logERROR) << "func_no can only be values in [2, 445]";
        return;
    }
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << input3 << ") failed.";
    }
}


void reverseList_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.reverseList(l1);
    if(list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}


void swapPairs_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.swapPairs(l1);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

void removeElements_scaffold(std::string input1, int val, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.removeElements(l1, val);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", val: " << val << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", val: " << val << ", " << expectedResult << ") failed";
    }
}

void deleteDuplicates_scaffold(std::string input1, std::string input2, int func_no) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = nullptr;
    if (func_no == 82) {
        ans = ss.deleteDuplicates_082(l1);
    } else if (func_no == 83) {
        ans = ss.deleteDuplicates_083(l1);
    } else {
        util::Log(logERROR) << "func_no can only be values in [82, 83]";
        return;
    }
    if (list_equal(ans, l2)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << func_no << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << func_no <<  ") failed. actual: ";
        printLinkedList(ans);
    }
}


void partition_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.partition(l1, x);
    if (list_equal(ans, l2)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << x << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << x <<  ") failed. actual: ";
        printLinkedList(ans);
    }
}


void removeNthFromEnd_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.removeNthFromEnd(l1, x);
    if (list_equal(ans, l2)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << x << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << x <<  ") failed. actual: ";
        printLinkedList(ans);
    }
}


void reverseKGroup_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.reverseKGroup(l1, x);
    if (list_equal(ans, l2)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << x << ") passed.";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << x <<  ") failed. actual: ";
        printLinkedList(ans);
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running reverseList tests:";
    TIMER_START(reverseList);
    reverseList_scaffold("[1,2,3,4,5]", "[5,4,3,2,1]");
    reverseList_scaffold("[1,2,3,4,]", "[4,3,2,1]");
    reverseList_scaffold("[1]", "[1]");
    reverseList_scaffold("[]", "[]");
    TIMER_STOP(reverseList);
    util::Log(logESSENTIAL) << "Running reverseList tests uses " << TIMER_MSEC(reverseList) << "ms.";

    util::Log(logESSENTIAL) << "Running swapPairs tests:";
    TIMER_START(swapPairs);
    swapPairs_scaffold("[1,2,3,4,5]", "[2,1,4,3,5]");
    swapPairs_scaffold("[1,2,3,4]", "[2,1,4,3]");
    swapPairs_scaffold("[1]", "[1]");
    swapPairs_scaffold("[]", "[]");
    TIMER_STOP(swapPairs);
    util::Log(logESSENTIAL) << "Running swapPairs tests uses " << TIMER_MSEC(swapPairs) << "ms.";

    util::Log(logESSENTIAL) << "Running addTwoNumbers tests:";
    TIMER_START(addTwoNumbers);
    addTwoNumbers_scaffold("[3,4,5]", "[7,0,8]", "[0,5,3,1]", 2);
    addTwoNumbers_scaffold("[2,4,3]", "[5,6,4]", "[7,0,8]", 2);
    addTwoNumbers_scaffold("[2,4,3]", "[6,6,3]", "[8,0,7]", 2);
    addTwoNumbers_scaffold("[1]", "[9]", "[0, 1]", 2);
    addTwoNumbers_scaffold("[1]", "[9,9,9]", "[0,0,0,1]", 2);
    addTwoNumbers_scaffold("[3,4,5]", "[7,0,8]", "[1,0,5,3]", 445);
    addTwoNumbers_scaffold("[2,4,3]", "[5,6,4]", "[8,0,7]", 445);
    addTwoNumbers_scaffold("[7,2,4,3]", "[5,6,4]", "[7,8,0,7]", 445);
    addTwoNumbers_scaffold("[2,4,3]", "[6,6,3]", "[9,0,6]", 445);
    addTwoNumbers_scaffold("[1]", "[9]", "[1, 0]", 445);
    addTwoNumbers_scaffold("[1]", "[9,9,9]", "[1,0,0,0]", 445);
    TIMER_STOP(addTwoNumbers);
    util::Log(logESSENTIAL) << "Running addTwoNumbers tests uses " << TIMER_MSEC(addTwoNumbers) << "ms.";

    util::Log(logESSENTIAL) << "Running removeElements tests:";
    TIMER_START(removeElements);
    removeElements_scaffold("[1,2,6,3,4,5,6]", 6, "[1,2,3,4,5]");
    removeElements_scaffold("[1,2,6,3,4,5,6]", 1, "[2,6,3,4,5,6]");
    removeElements_scaffold("[1,2,6,3,4,5,6]", 8, "[1,2,6,3,4,5,6]");
    removeElements_scaffold("[]", 6, "[]");
    removeElements_scaffold("[6,6,6,6]", 6, "[]");
    TIMER_STOP(swapPairs);
    util::Log(logESSENTIAL) << "Running removeElements tests uses " << TIMER_MSEC(removeElements) << "ms.";

    util::Log(logESSENTIAL) << "Running deleteDuplicates tests:";
    TIMER_START(deleteDuplicates);
    deleteDuplicates_scaffold("[1,1,1]", "[]", 82);
    deleteDuplicates_scaffold("[1,1,2]", "[2]", 82);
    deleteDuplicates_scaffold("[1,2,2,3]", "[1,3]", 82);
    deleteDuplicates_scaffold("[1,1,2,3,3]", "[2]", 82);
    deleteDuplicates_scaffold("[1,2,3,3,4,4,5]", "[1,2,5]", 82);
    deleteDuplicates_scaffold("[1,1,1]", "[]", 82);
    deleteDuplicates_scaffold("[1,1,1]", "[1]", 83);
    deleteDuplicates_scaffold("[1,1,2]", "[1,2]", 83);
    deleteDuplicates_scaffold("[1,2,2,3]", "[1,2,3]", 83);
    deleteDuplicates_scaffold("[1,1,2,3,3]", "[1,2,3]", 83);
    deleteDuplicates_scaffold("[1,2,3,3,4,4,5]", "[1,2,3,4,5]", 83);
    TIMER_STOP(addTwoNumbers);
    util::Log(logESSENTIAL) << "Running deleteDuplicates tests uses " << TIMER_MSEC(deleteDuplicates) << "ms.";

    util::Log(logESSENTIAL) << "Running partition tests:";
    TIMER_START(partition);
    partition_scaffold("[1,1,1]", "[1,1,1]", 82);
    partition_scaffold("[1,1,1]", "[1,1,1]", 0);
    partition_scaffold("[1,1,2]", "[1,1,2]", 2);
    partition_scaffold("[1,4,3,2,5,2]", "[1,2,2,4,3,5]", 3);
    partition_scaffold("[2,1]", "[1,2]", 2);
    TIMER_STOP(partition);
    util::Log(logESSENTIAL) << "Running partition tests uses " << TIMER_MSEC(partition) << "ms.";

    util::Log(logESSENTIAL) << "Running removeNthFromEnd tests:";
    TIMER_START(removeNthFromEnd);
    removeNthFromEnd_scaffold("[1,2,3,4,5]", "[1,2,3,5]", 2);
    removeNthFromEnd_scaffold("[1]", "[]", 1);
    removeNthFromEnd_scaffold("[1,2]", "[1]", 1);
    removeNthFromEnd_scaffold("[1,2]", "[2]", 2);
    removeNthFromEnd_scaffold("[1,2,3]", "[1,2,3]", 4);
    TIMER_STOP(removeNthFromEnd);
    util::Log(logESSENTIAL) << "Running removeNthFromEnd tests uses " << TIMER_MSEC(removeNthFromEnd) << "ms.";

    util::Log(logESSENTIAL) << "Running reverseKGroup tests:";
    TIMER_START(reverseKGroup);
    reverseKGroup_scaffold("[1,2,3,4,5]", "[2,1,4,3,5]", 2);
    reverseKGroup_scaffold("[1,2,3,4,5]", "[3,2,1,4,5]", 3);
    reverseKGroup_scaffold("[1]", "[1]", 1);
    reverseKGroup_scaffold("[1,2]", "[1,2]", 1);
    reverseKGroup_scaffold("[1,2]", "[2,1]", 2);
    TIMER_STOP(reverseKGroup);
    util::Log(logESSENTIAL) << "Running reverseKGroup tests uses " << TIMER_MSEC(reverseKGroup) << "ms.";
}
