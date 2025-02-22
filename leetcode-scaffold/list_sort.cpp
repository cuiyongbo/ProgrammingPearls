#include "leetcode.h"

using namespace std;

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
        if (t != nullptr) {
            pq.push(t);
        }
    }
    while (!pq.empty()) {
        auto t = pq.top(); pq.pop();
        p->next = t; p = p->next; // push_back
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
        ListNode* tmp = head->next;
        ListNode* q = dummy.next;
        ListNode* p = &dummy; // previous node of q
        while (q != nullptr) {
            // head should be inserted before q
            if (head->val < q->val) {
                break;
            }
            p = q; // update p as the previous node of q->next
            q = q->next;
        }
        p->next = head; head->next = q;
        head = tmp;
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
    ListNode* fast = head;
    ListNode* slow = head;
    ListNode* p = slow;
    while (fast != nullptr) {
        fast = fast->next;
        if (fast != nullptr) {
            fast = fast->next;
        }
        p = slow;
        slow = slow->next;
    }
    p->next = nullptr; // IMPORTANT: split original list into two pieces
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
        SPDLOG_INFO("Case({}, {}, {}) passed", input1, input2, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}, {}) failed", input1, input2, expectedResult);
    }
}


void mergeKLists_scaffold(std::string input, std::string expectedResult) {
    vector<vector<int>> arrs = stringTo2DArray<int>(input);
    std::vector<ListNode*> lists;
    lists.reserve(arrs.size());
    for (auto& s: arrs) {
        lists.push_back(vectorToListNode(s));
    }
    Solution ss;
    ListNode* ans = ss.mergeKLists(lists);
    ListNode* expected = stringToListNode(expectedResult);
    if (list_equal(ans, expected)) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed", input, expectedResult);
    }
}


void insertionSortList_scaffold(std::string input, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.insertionSortList(l1);
    if (list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed", input, expectedResult);
    }
}


void sortList_scaffold(std::string input, std::string expectedResult) {
    ListNode* l1 = stringToListNode(input);
    ListNode* l3 = stringToListNode(expectedResult);
    Solution ss;
    ListNode* ans = ss.sortList(l1);
    if (list_equal(ans, l3)) {
        SPDLOG_INFO("Case({}, {}) passed", input, expectedResult);
    } else {
        SPDLOG_ERROR("Case({}, {}) failed", input, expectedResult);
    }
}


void verbose_sortList_tests(int array_size) {
    //std::random_device rd;
    //std::mt19937 g(rd());
    std::mt19937 g(12345); // for reproducibility
    vector<int> vi;
    generateTestArray(vi, array_size, false, false);
    for (int i=0; i<1000; i++) {
        int n = rand() % array_size;
        std::shuffle(vi.begin(), vi.begin()+n, g);
        Solution ss;
        ListNode* l1 = vectorToListNode(vi);
        ListNode* ans = ss.sortList(l1);
        std::sort(vi.begin(), vi.end(), std::less<int>());
        ListNode* l2 = vectorToListNode(vi);
        if (list_equal(ans, l2)) {
            //SPDLOG_INFO("Case(array_size={}, shuffle={}) passed", array_size, n);
        } else {
            SPDLOG_ERROR("Case(array_size={}, shuffle={}) failed", array_size, n);
        }
    }
}


int main(int argc, char* argv[]) {
    SPDLOG_WARN("Running mergeTwoLists tests:");
    TIMER_START(mergeTwoLists);
    mergeTwoLists_scaffold("[]", "[]", "[]");
    mergeTwoLists_scaffold("[1]", "[2,3]", "[1,2,3]");
    mergeTwoLists_scaffold("[1,2,3,4,5]", "[]", "[1,2,3,4,5]");
    mergeTwoLists_scaffold("[1,2,3,4,5,6,7,8,9,10]", "[5,6]", "[1,2,3,4,5,5,6,6,7,8,9,10]");
    TIMER_STOP(mergeTwoLists);
    SPDLOG_WARN("mergeTwoLists using {} ms", TIMER_MSEC(mergeTwoLists));

    SPDLOG_WARN("Running mergeKLists tests:");
    TIMER_START(mergeKLists);
    mergeKLists_scaffold("[]", "[]");
    mergeKLists_scaffold("[[1],[2,3]]", "[1,2,3]");
    mergeKLists_scaffold("[[1],[2,3],[],[1,2,3,4,5]]", "[1,1,2,2,3,3,4,5]");
    TIMER_STOP(mergeKLists);
    SPDLOG_WARN("mergeKLists using {} ms", TIMER_MSEC(mergeKLists));

    SPDLOG_WARN("Running insertionSortList tests:");
    TIMER_START(insertionSortList);
    insertionSortList_scaffold("[3,1,2]", "[1,2,3]");
    insertionSortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    insertionSortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    insertionSortList_scaffold("[2,2]", "[2,2]");
    insertionSortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(insertionSortList);
    SPDLOG_WARN("insertionSortList using {} ms", TIMER_MSEC(insertionSortList));

    int array_size = 1000;
    if (argc > 1) {
        array_size = std::atoi(argv[1]);
        if (array_size <= 0) {
            SPDLOG_WARN("Usage: {} [arrary_size]", argv[0]);
            SPDLOG_WARN("\tarrary_size must be positive, default to 100 if unspecified");
            return -1;
        }
    }

    SPDLOG_WARN("Running mergeSortList tests:");
    TIMER_START(mergeSortList);
    sortList_scaffold("[3,1,2]", "[1,2,3]");
    sortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    sortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    sortList_scaffold("[2,2]", "[2,2]");
    sortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    verbose_sortList_tests(array_size);
    TIMER_STOP(mergeSortList);
    SPDLOG_WARN("mergeSortList using {} ms", TIMER_MSEC(mergeSortList));
}
