#include "leetcode.h"

using namespace std;
using namespace osrm;
using namespace osrm::timing;

/* leetcode: 21, 23, 147, 148 */

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    ListNode* mergeKLists(vector<ListNode*>& lists);
    ListNode* insertionSortList(ListNode* head);
    ListNode* sortList(ListNode* head);

private:
    ListNode* mergeKLists_minHeap(vector<ListNode*>& lists);
    ListNode* mergeSort(ListNode* head, ListNode* tail);
};

ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2) {
/*
    Merge Two Sorted Lists    
*/
    ListNode dummy;
    ListNode* p = &dummy;
    while (l1 != nullptr && l2 != nullptr) {
        if (l1->val > l2->val) {
            p->next = l2;
            p = p->next;
            l2 = l2->next;
        } else {
            p->next = l1;
            p = p->next;
            l1 = l1->next;
        }
    }
    p->next = (l1 != nullptr) ? l1 : l2;
    return dummy.next;
}

ListNode* Solution::mergeKLists(vector<ListNode*>& lists) {
/*
    Merge K sorted lists
*/
    function<ListNode*(int, int)> divide_and_conquer = [&] (int l, int r) {
        if (l > r) {
            return (ListNode*)nullptr;
        } else if (l == r) {
            return lists[l];
        } else {
            int mid = (l+r)/2;
            ListNode* left = divide_and_conquer(l, mid);
            ListNode* right = divide_and_conquer(mid+1, r);
            return mergeTwoLists(left, right);
        }
    };
    return divide_and_conquer(0, lists.size()-1);

    //return mergeKLists_minHeap(lists);
}

ListNode* Solution::mergeKLists_minHeap(vector<ListNode*>& lists)
{
    auto cmp = [](ListNode* l, ListNode* r) { return l->val > r->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    for(auto l: lists)
    {
        while(l != nullptr)
        {
            pq.push(l);
            l = l->next;
        }
    }

    ListNode dummy;
    ListNode* p = &dummy;
    while(!pq.empty())
    {
        auto t = pq.top(); pq.pop();
        p->next = t;
        p = p->next;
    }
    return dummy.next;
}

ListNode* Solution::insertionSortList(ListNode* head) {
    ListNode dummy(INT32_MIN);
    while (head != nullptr) {
        ListNode* p = &dummy;
        ListNode* q = p;
        // stable sort
        while (p != nullptr && p->val <= head->val) {
            q = p;
            p = p->next;
        }
        ListNode* t = head->next;
        q->next = head;
        head->next = p;
        head = t;
    }
    return dummy.next;
}

ListNode* Solution::sortList(ListNode* head) {
/*
    Given the head of a linked list, return the list after sorting it in ascending order.
    Follow up: Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?
*/
    return mergeSort(head, nullptr);
}

ListNode* Solution::mergeSort(ListNode* head, ListNode* tail) {
    if(head == nullptr) {
        return head;
    }

    if(head->next == tail) {
        // make sure that the trivial case returns an isolated node
        head->next = nullptr;
        return head;
    }

    ListNode* p = head;
    ListNode* mid = head;
    while (p != tail) {
        p = p->next;
        if (p == tail) {
            break;
        }
        p = p->next;
        mid = mid->next;
    }

    ListNode* left = mergeSort(head, mid);
    ListNode* right = mergeSort(mid, tail);
    return mergeTwoLists(left, right);
}

void mergeTwoLists_scaffold(string input1, string input2, string expectedResult) {
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

void mergeKLists_scaffold(vector<string>& input, string expectedResult) {
    vector<ListNode*> lists;
    lists.reserve(input.size());
    for (auto& s: input) {
        lists.push_back(stringToListNode(s));
    }

    Solution ss;
    ListNode* ans = ss.mergeKLists(lists);
    ListNode* expected = stringToListNode(expectedResult);
    if(list_equal(ans, expected)) {
        util::Log(logINFO) << "Case(" << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << expectedResult << ") failed";
    }
}

void insertionSortList_scaffold(string input1, string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.insertionSortList(l1);
    if(list_equal(ans, l3)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

void sortList_scaffold(string input1, string expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.sortList(l1);
    if(list_equal(ans, l3)) {
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

    vector<string> input;
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
    util::Log(logESSENTIAL) << "Running sortList tests:";
    sortList_scaffold("[3,1,2]", "[1,2,3]");
    sortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    sortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    sortList_scaffold("[2,2]", "[2,2]");
    sortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(mergeSortList);
    util::Log(logESSENTIAL) << "mergeSortList: " << TIMER_MSEC(mergeSortList) << " milliseconds.";
}
