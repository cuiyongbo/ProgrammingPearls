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
{
    ListNode dummy;
    ListNode* p = &dummy;
    while (head != nullptr) {
        ListNode* tmp = head->next;
        head->next = nullptr; // DEBUG_ONLY
        head->next = p->next; p->next = head; // push_front
        head = tmp;
    }
    return dummy.next;
}

}


ListNode* Solution::reverseKGroup(ListNode* head, int k) {
/*
Given the head of a linked list, reverse the nodes of the list by k nodes at a time, and return the modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
You may not alter the values in the list's nodes, only nodes themselves may be changed.

Examples:
    Input: head = [1,2,3,4,5], k = 2
    Output: [2,1,4,3,5]
    Input: head = [1,2,3,4,5], k = 3
    Output: [3,2,1,4,5]
*/

{
    ListNode dummy;
    ListNode* p = &dummy;
    std::stack<ListNode*> st;
    while (head != nullptr) {
        ListNode* tmp = head->next; 
        // DON'T waste your mind, you have to isolate node before pushing it to stack
        head->next = nullptr;
        st.push(head);
        // reverse nodes in a k-subgroup
        if ((int)st.size() == k) {
            while (!st.empty()) {
                auto node = st.top(); st.pop();
                p->next = node; p = p->next; // push_backss
            }
        }
        head = tmp;
    }
    // for left-out nodes, we should remain their original order
    while (!st.empty()) {
        auto node = st.top(); st.pop();
        node->next = p->next; p->next = node; // push_front
    }
    return dummy.next;
}

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
        ListNode* tmp = head->next;
        head->next = nullptr; // isolate nodes
        st.push(head);
        head = tmp;
    }
    ListNode dummy;
    ListNode* p = &dummy;
    int index = 0;
    while (!st.empty()) {
        auto t = st.top(); st.pop();
        index++;
        if (index == n) {
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
        ListNode* tmp = head->next; head->next = nullptr;
        if (head->val != val) {
            p->next = head; p = p->next; // push_back
        }
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
    You are given two non-empty linked lists representing two *non-negative* integers. 
    The digits are stored in *LSB-first* order and each of their nodes contain a single digit. 
    Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Example:
        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 0 -> 8
        Explanation: 342 + 465 = 807.
*/

{
    int carry = 0;
    ListNode dummy(-1);
    ListNode* p = &dummy;
    ListNode* p1 = l1;
    ListNode* p2 = l2;
    while (p1 != nullptr || p2 != nullptr || carry != 0) {
        int val = 0;
        if (p1 != nullptr) {
            val += p1->val;
            p1 = p1->next;
        }
        if (p2 != nullptr) {
            val += p2->val;
            p2 = p2->next;
        }       
        val += carry;
        if (val >= 10) {
            val -= 10;
            carry = 1;
        } else {
            carry = 0;
        }
        ListNode* node = new ListNode(val);
        p->next = node; p = p->next; // push_back
    }
    return dummy.next;
}

}


ListNode* Solution::addTwoNumber_445(ListNode* l1, ListNode* l2) {
/*
    You are given two non-empty linked lists representing two *non-negative* integers. 
    The digits are stored in *MSB-first* order and each of their nodes contain a single digit. 
    Add the two numbers and return it as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    Example:
        Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 8 -> 0 -> 7
*/

{
    std::stack<int> st1;
    std::stack<int> st2;
    ListNode* p1 = l1;
    ListNode* p2 = l2;
    while (p1 != nullptr || p2 != nullptr) {
        if (p1 != nullptr) {
            st1.push(p1->val);
            p1 = p1->next;
        }
        if (p2 != nullptr) {
            st2.push(p2->val);
            p2 = p2->next;
        }
    }

    int carry = 0;
    ListNode dummy(-1);
    ListNode* p = &dummy;
    while (!st1.empty() || !st2.empty() || carry != 0) {
        int val = 0;
        if (!st1.empty()) {
            val += st1.top(); st1.pop();
        }
        if (!st2.empty()) {
            val += st2.top(); st2.pop();
        }    
        val += carry;
        if (val >= 10) {
            val -= 10;
            carry = 1;
        } else {
            carry = 0;
        }
        ListNode* node = new ListNode(val);
        node->next = p->next; p->next = node; // push_front
    }
    return dummy.next;
}

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

    // 2. you can image there is a virtual null node connecting the two lists, then iterate over them as usual
{
    ListNode* p1 = l1;
    ListNode* p2 = l2;
    while (p1 != p2) {
        p1 = p1->next;
        if (p1 == nullptr) {
            p1 = l2;
        }
        p2 = p2->next;
        if (p2 == nullptr) {
            p2 = l1;
        }
    }
    return p1;
}

{
    ListNode* p1 = l1;
    ListNode* p2 = l2;
    while (p1 != p2) {
        p1 = p1==nullptr ? l2 : p1->next;
        p2 = p2==nullptr ? l1 : p2->next;
    }
    return p1;
}

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
        SPDLOG_ERROR("func_no can only be values in [2, 445], actual: {}", func_no);
        return;
    }
    if (list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}, {}) passed.",  input1, input2, input3);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed.",  input1, input2, input3);
    }
}


void reverseList_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.reverseList(l1);
    if(list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed", input1, expectedResult);
    }
}


void swapPairs_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.swapPairs(l1);
    if (list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}) passed", input1, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed", input1, expectedResult);
    }
}


void reverseKGroup_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.reverseKGroup(l1, x);
    if (list_equal(ans, l2)) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, input2, x);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed. acutal:", input1, input2, x);
        printLinkedList(ans);
    }
}


void removeElements_scaffold(std::string input1, int val, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.removeElements(l1, val);
    if (list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, val, expectedResult);
    } else {
        SPDLOG_INFO("Case({}, {}, {}) failed", input1, val, expectedResult);
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
        SPDLOG_ERROR("func_no can only be values in [82, 83], actual: {}", func_no);
        return;
    }
    if (list_equal(ans, l2)) {
        SPDLOG_INFO("Case({}, {}, {}) passed.",  input1, input2, func_no);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed. actual:",  input1, input2, func_no);
        printLinkedList(ans);
    }
}


void partition_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.partition(l1, x);
    if (list_equal(ans, l2)) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, input2, x);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed. acutal:", input1, input2, x);
        printLinkedList(ans);
    }
}


void removeNthFromEnd_scaffold(std::string input1, std::string input2, int x) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    Solution ss;
    ListNode* ans = ss.removeNthFromEnd(l1, x);
    if (list_equal(ans, l2)) {
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, input2, x);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed. acutal:", input1, input2, x);
        printLinkedList(ans);
    }
}


int main() {
    SPDLOG_WARN("Running reverseList tests:");
    TIMER_START(reverseList);
    reverseList_scaffold("[1,2,3,4,5]", "[5,4,3,2,1]");
    reverseList_scaffold("[2,8,9,3,6,9,3,4,1,7,4]", "[4,7,1,4,3,9,6,3,9,8,2]");
    reverseList_scaffold("[2,0,4,8,5,2,8,1]", "[1,8,2,5,8,4,0,2]");
    reverseList_scaffold("[6,7,5,2,9,1,5,5,9,8,2]", "[2,8,9,5,5,1,9,2,5,7,6]");
    reverseList_scaffold("[1,2,3,4,]", "[4,3,2,1]");
    reverseList_scaffold("[1]", "[1]");
    reverseList_scaffold("[]", "[]");
    TIMER_STOP(reverseList);
    SPDLOG_WARN("Running reverseList tests uses {} ms", TIMER_MSEC(reverseList));

    SPDLOG_WARN("Running swapPairs tests:");
    TIMER_START(swapPairs);
    swapPairs_scaffold("[1,2,3,4,5]", "[2,1,4,3,5]");
    swapPairs_scaffold("[1,2,3,4]", "[2,1,4,3]");
    swapPairs_scaffold("[1]", "[1]");
    swapPairs_scaffold("[]", "[]");
    TIMER_STOP(swapPairs);
    SPDLOG_WARN("Running swapPairs tests uses {} ms", TIMER_MSEC(swapPairs));

    SPDLOG_WARN("Running reverseKGroup tests:");
    TIMER_START(reverseKGroup);
    reverseKGroup_scaffold("[1,2,3,4,5]", "[2,1,4,3,5]", 2);
    reverseKGroup_scaffold("[1,2,3,4,5]", "[3,2,1,4,5]", 3);
    reverseKGroup_scaffold("[1]", "[1]", 1);
    reverseKGroup_scaffold("[1,2]", "[1,2]", 1);
    reverseKGroup_scaffold("[1,2]", "[2,1]", 2);
    TIMER_STOP(reverseKGroup);
    SPDLOG_WARN("Running reverseKGroup tests uses {} ms", TIMER_MSEC(reverseKGroup));

    SPDLOG_WARN("Running addTwoNumbers tests:");
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
    addTwoNumbers_scaffold("[2,8,9,3,6,9,3,4,1,7,4]", "[1,8,2,5,8,4,0,2]", "[2,8,9,5,5,1,9,2,5,7,6]", 445);
    addTwoNumbers_scaffold("[2,0,4,8,5,2,8,1]", "[4,7,1,4,3,9,6,3,9,8,2]", "[6,7,5,2,9,1,5,5,9,8,2]", 2);
    TIMER_STOP(addTwoNumbers);
    SPDLOG_WARN("Running addTwoNumbers tests uses {} ms", TIMER_MSEC(addTwoNumbers));

    SPDLOG_WARN("Running removeElements tests:");
    TIMER_START(removeElements);
    removeElements_scaffold("[1,2,6,3,4,5,6]", 6, "[1,2,3,4,5]");
    removeElements_scaffold("[1,2,6,3,4,5,6]", 1, "[2,6,3,4,5,6]");
    removeElements_scaffold("[1,2,6,3,4,5,6]", 8, "[1,2,6,3,4,5,6]");
    removeElements_scaffold("[1,2,6,3,4,5,6]", 3, "[1,2,6,4,5,6]");
    removeElements_scaffold("[]", 6, "[]");
    removeElements_scaffold("[6,6,6,6]", 6, "[]");
    TIMER_STOP(removeElements);
    SPDLOG_WARN("Running removeElements tests uses {} ms", TIMER_MSEC(removeElements));

    SPDLOG_WARN("Running deleteDuplicates tests:");
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
    TIMER_STOP(deleteDuplicates);
    SPDLOG_WARN("Running deleteDuplicates tests uses {} ms", TIMER_MSEC(deleteDuplicates));

    SPDLOG_WARN("Running partition tests:");
    TIMER_START(partition);
    partition_scaffold("[1,1,1]", "[1,1,1]", 82); // left partition is empty
    partition_scaffold("[1,1,1]", "[1,1,1]", 0); // right partition is empty
    partition_scaffold("[1,1,2]", "[1,1,2]", 2); // no need to change the array 
    partition_scaffold("[1,4,3,2,5,2]", "[1,2,2,4,3,5]", 3);
    partition_scaffold("[2,1]", "[1,2]", 2);
    TIMER_STOP(partition);
    SPDLOG_WARN("Running partition tests uses {} ms", TIMER_MSEC(partition));

    SPDLOG_WARN("Running removeNthFromEnd tests:");
    TIMER_START(removeNthFromEnd);
    removeNthFromEnd_scaffold("[1,2,3,4,5]", "[1,2,3,5]", 2);
    removeNthFromEnd_scaffold("[1]", "[]", 1);
    removeNthFromEnd_scaffold("[1,2]", "[1]", 1);
    removeNthFromEnd_scaffold("[1,2]", "[2]", 2);
    removeNthFromEnd_scaffold("[1,2,3]", "[1,2,3]", 4);
    TIMER_STOP(removeNthFromEnd);
    SPDLOG_WARN("Running removeNthFromEnd tests uses {} ms", TIMER_MSEC(removeNthFromEnd));

}
