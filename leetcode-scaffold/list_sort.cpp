#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 21, 23, 147, 148 */

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    ListNode* mergeKLists(std::vector<ListNode*>& lists);
    ListNode* insertionSortList(ListNode* head);
    ListNode* sortList(ListNode* head);
};

ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2) {
/*
    Merge two sorted linked lists and return it as a new sorted list. 
    The new list should be made by splicing together the nodes of the first two lists. 
*/
    ListNode dummy;
    ListNode* p = &dummy;
    while (l1 != nullptr && l2 != nullptr) {
        if (l1->val < l2->val) {
            p->next = l1; p = p->next; // push_back
            l1 = l1->next;
        } else {
            p->next = l2; p = p->next;
            l2 = l2->next;
        }
    }
    p->next = l1 != nullptr ? l1 : l2;
    return dummy.next;
}

ListNode* Solution::mergeKLists(std::vector<ListNode*>& lists) {
/*
    Merge k sorted linked lists and return it as one sorted list.
*/
    ListNode dummy;
    ListNode* p = &dummy;
    // min-heap ordered by ListNode::val
    auto cmp = [] (ListNode* a, ListNode* b) {
        return a->val > b->val;
    };
    std::priority_queue<ListNode*, std::vector<ListNode*>, decltype(cmp)> pq(cmp);
    for (auto t: lists) {
        if (t!=nullptr) {
            pq.push(t);
        }
    }
    while (!pq.empty()) {
        auto t = pq.top(); pq.pop();
        p->next = t; p = p->next;
        if (t->next != nullptr) {
            pq.push(t->next);
        }
    }
    return dummy.next;
}


ListNode* Solution::insertionSortList(ListNode* head) {
/*
    Sort a linked list using insertion sort.
*/
    ListNode dummy;
    while (head != nullptr) {
        ListNode* p = &dummy;
        ListNode* q = p->next;
        while (q != nullptr && q->val <= head->val) {
            p = q;
            q = q->next;
        }
        ListNode* tmp = head->next;
        head->next = nullptr; // cut head out from original list
        p->next = head; // insert head between p and q
        head->next = q;
        head = tmp; // update head
    }
    return dummy.next;
}


ListNode* Solution::sortList(ListNode* head) {
/*
    Given the head of a linked list, return the list after sorting it in ascending order.
    Follow up: Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?
    Hint: perform mergeSort on the list
    For example, given an input [3,1,2], the path of execution is:
        dac(3, null)
            mid = 1
            dac(3,1) -> [3,null]
            dac(1,null)
                mid = 2
                dac(1,2) -> [1, null]
                dac(2,null) -> [2, null]
                merger: 1->2->null
            merger: 1->2->3->null
*/
    if (head == nullptr || head->next == nullptr) { // trivial case
        return head;
    }
    // List contains two nodes at least
    ListNode* fast = head;
    ListNode* slow = head;
    ListNode* p = slow;
    while (fast != nullptr) {
        fast = fast->next;
        if (fast == nullptr) {
            break;
        }
        fast = fast->next;
        p = slow;
        slow = slow->next;
    }
    p->next = nullptr; // divide List into two pieces
    ListNode* l = sortList(head);
    ListNode* r = sortList(slow);
    return mergeTwoLists(l, r);
}


void mergeTwoLists_scaffold(std::string input1, std::string input2, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);

    Solution ss;
    ListNode* ans = ss.mergeTwoLists(l1, l2);
    ListNode* expected = stringToListNode(expectedResult);
    if (list_equal(ans, expected)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed";
    }
}

void mergeKLists_scaffold(std::vector<string>& input, std::string expectedResult) {
    std::vector<ListNode*> lists;
    lists.reserve(input.size());
    for (auto& s: input) {
        lists.push_back(stringToListNode(s));
    }

    Solution ss;
    ListNode* ans = ss.mergeKLists(lists);
    ListNode* expected = stringToListNode(expectedResult);
    if (list_equal(ans, expected)) {
        util::Log(logINFO) << "Case(" << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << expectedResult << ") failed";
    }
}

void insertionSortList_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.insertionSortList(l1);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

void sortList_scaffold(std::string input1, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.sortList(l1);
    if (list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running mergeTwoLists tests:";
    TIMER_START(mergeTwoLists);
    mergeTwoLists_scaffold("[]", "[]", "[]");
    mergeTwoLists_scaffold("[1]", "[2,3]", "[1,2,3]");
    mergeTwoLists_scaffold("[1,2,3,4,5]", "[]", "[1,2,3,4,5]");
    mergeTwoLists_scaffold("[1,2,3,4,5,6,7,8,9,10]", "[5,6]", "[1,2,3,4,5,5,6,6,7,8,9,10]");
    TIMER_STOP(mergeTwoLists);
    util::Log(logESSENTIAL) << "mergeTwoLists: " << TIMER_MSEC(mergeTwoLists) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running mergeKLists tests:";
    TIMER_START(mergeKLists);

    std::vector<string> input;
    mergeKLists_scaffold(input, "[]");

    input.clear();
    input.push_back("[1]");
    input.push_back("[2,3]");
    mergeKLists_scaffold(input, "[1,2,3]");

    input.clear();
    input.push_back("[1]");
    input.push_back("[2,3]");
    input.push_back("[]");
    input.push_back("[1,2,3,4,5]");
    mergeKLists_scaffold(input, "[1,1,2,2,3,3,4,5]");

    TIMER_STOP(mergeKLists);
    util::Log(logESSENTIAL) << "mergeKLists: " << TIMER_MSEC(mergeKLists) << " milliseconds.";

    TIMER_START(insertionSortList);
    util::Log(logESSENTIAL) << "Running insertionSortList tests:";
    insertionSortList_scaffold("[3,1,2]", "[1,2,3]");
    insertionSortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    insertionSortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    insertionSortList_scaffold("[2,2]", "[2,2]");
    insertionSortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(insertionSortList);
    util::Log(logESSENTIAL) << "insertionSortList: " << TIMER_MSEC(insertionSortList) << " milliseconds.";

    TIMER_START(mergeSortList);
    util::Log(logESSENTIAL) << "Running mergeSortList tests:";
    sortList_scaffold("[3,1,2]", "[1,2,3]");
    sortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    sortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    sortList_scaffold("[2,2]", "[2,2]");
    sortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(mergeSortList);
    util::Log(logESSENTIAL) << "mergeSortList: " << TIMER_MSEC(mergeSortList) << " milliseconds.";
}
