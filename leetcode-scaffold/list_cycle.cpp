#include "leetcode.h"

using namespace std;

/* leetcode: 141, 142 */

class Solution {
public:
    bool hasCycle(ListNode* head);
    ListNode* detectCycle(ListNode* head);
};

bool Solution::hasCycle(ListNode* head) {
/*
    Given a linked list, determine if it has a cycle in it.

    To represent a cycle in the given linked list, we use an integer pos which 
    represents the position (0-indexed) in the linked list where tail connects to. 
    If pos is -1, then there is no cycle in the linked list.
*/
    ListNode* fast = head;
    ListNode* slow = head;
    while (fast != nullptr) {
        fast = fast->next;
        if (fast == nullptr) {
            return false;
        }
        fast = fast->next; // fast pointer travels at speed 2
        slow = slow->next; // slow pointer travels at speed 1
        // if there is cycle in the list, fast, slow would meet at certain position on the cycle
        if (slow == fast) {
            return true;
        }
    }
    return false;
}

ListNode* Solution::detectCycle(ListNode* head) {
/* 
    Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
*/
    // 1. check whether there is a cycle in the list or not
    ListNode* fast = head;
    ListNode* slow = head;
    while (fast != nullptr) {
        fast = fast->next;
        if (fast == nullptr) {
            break;
        }
        fast = fast->next;
        slow = slow->next;
        if (slow == fast) {
            break;
        }
    }
    // there is no cycle in the list
    if (fast == nullptr) {
        return nullptr;
    }
    // 2. if there is, find the node where the cycle begins 
    // PS: try to prove why fast and slow would rendezvous at the starting node
    /*
        x = distance(head, intersection)
        r = distance(intersection, p)
        p: rendezvous point on the cycle
        s: perimeter of the cycle
        fast: x + n*s + r = 2t
        slow: x + m*s + r = t
        2*(x+m*s+r) = x+n*s+r
        x+2ms+2r = n*s + r
        x + r = (n-2m)s *IMPORTANT*
        so after fast, slow rendezvous at p, we set fast to re-traverse from head at speed 1, and slow also continue to traverse from p at speed 1, they would rendezvous at intersection
    */

    fast = head;
    while (fast != slow) {
        fast = fast->next;
        slow = slow->next;
    }
    return fast;
}

void hasCycle_scaffold(const std::vector<int>& input1, int pos) {
    ListNode* l1 = vectorToListNode(input1);
    ListNode* p = nullptr;
    ListNode* q = nullptr;
    ListNode* head = l1;
    int i = 0;
    while (head != nullptr) {
        p = head;
        if (i++ == pos) {
            q = head;
        }
        head = head->next;
    }
    if (p != nullptr) {
        p->next = q;
    }
    Solution ss;
    bool expectedResult = pos>=0 && pos<input1.size();
    bool actual = ss.hasCycle(l1);
    if (actual == expectedResult) {
        SPDLOG_INFO("Case(list_size={}, pos={}) passed", input1.size(), pos);
    } else {
        SPDLOG_ERROR("Case(list_size={}, pos={}) failed", input1.size(), pos);
    }
}

void hasCycle_scaffold(std::string input1, int pos) {
    std::vector<int> vi = stringTo1DArray<int>(input1);
    hasCycle_scaffold(vi, pos);
}

void detectCycle_scaffold(const std::vector<int>& input1, int pos) {
    ListNode* l1 = vectorToListNode(input1);
    ListNode* p = nullptr;
    ListNode* q = nullptr; // intersection
    ListNode* head = l1;
    int i = 0;
    while (head != nullptr) {
        p = head;
        if (i++ == pos) {
            q = head;
        }
        head = head->next;
    }
    if (p != nullptr) {
        p->next = q;
    }
    Solution ss;
    ListNode* ans = ss.detectCycle(l1);
    if (q == ans) {
        SPDLOG_INFO("Case(list_size={}, pos={}) passed", input1.size(), pos);
    } else {
        SPDLOG_ERROR("Case(list_size={}, pos={}) failed", input1.size(), pos);
        if (q != nullptr) {
            SPDLOG_ERROR("expect {}", q->val);
        } else {
            SPDLOG_ERROR("expect null");
        }
        if (ans != nullptr) {
            SPDLOG_ERROR("actual {}", ans->val);
        } else {
            SPDLOG_ERROR("actual null");
        }
    }
}

void detectCycle_scaffold(std::string input1, int pos) {
    std::vector<int> vi = stringTo1DArray<int>(input1);
    detectCycle_scaffold(vi, pos);
}

void basic_test() {
    SPDLOG_WARN("Running hasCycle tests:");
    TIMER_START(hasCycle);
    hasCycle_scaffold("[]", -1);
    hasCycle_scaffold("[1]", -1);
    hasCycle_scaffold("[1,2,3,4]", -1);
    hasCycle_scaffold("[1,2,3,4,5]", -1);
    hasCycle_scaffold("[1]", 0);
    hasCycle_scaffold("[1,2,3,4,5]", 0);
    hasCycle_scaffold("[1,2,3,4]", 2);
    hasCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5);
    TIMER_STOP(hasCycle);
    SPDLOG_WARN("hasCycle using {} ms", TIMER_MSEC(hasCycle));

    SPDLOG_WARN("Running detectCycle tests:");
    TIMER_START(detectCycle);
    detectCycle_scaffold("[]", -1);
    detectCycle_scaffold("[1]", -1);
    detectCycle_scaffold("[1]", 0);
    detectCycle_scaffold("[1,2,3,4,5]", 2);
    detectCycle_scaffold("[1,2,3,4]", 2);
    detectCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5);
    TIMER_STOP(detectCycle);
    SPDLOG_WARN("detectCycle using {} ms", TIMER_MSEC(detectCycle));
}

void detectCycle_batch_test(int test_array_scale) {
    int pos = -1;
    vector<int> vi; vi.reserve(test_array_scale);
    for (int i=0; i<100; ++i) {
        vi.clear();
        int n = rand() % test_array_scale + 1;
        for (int j=0; j<n; ++j) {
            vi.push_back(j);
        }
        pos = rand() % (2*n + 1);
        //SPDLOG_INFO("detectCycle_batch_test(array_size={}, pos={})", n, pos);
        detectCycle_scaffold(vi, pos);
        hasCycle_scaffold(vi, pos);
    }
}

int main(int argc, char* argv[]) {
    basic_test();

    int test_array_scale = 100;
    if (argc > 1) {
        test_array_scale = std::atoi(argv[1]);
        if (test_array_scale <= 0) {
            cout << "test_array_scale must be positive, default to 100 if unspecified" << endl;
            return -1;
        }
    }

    SPDLOG_WARN("Running detectCycle_batch_test tests:");
    TIMER_START(detectCycle_batch_test);
    detectCycle_batch_test(test_array_scale);
    TIMER_STOP(detectCycle_batch_test);
    SPDLOG_WARN("detectCycle_batch_test using {} ms", TIMER_MSEC(detectCycle_batch_test));
}