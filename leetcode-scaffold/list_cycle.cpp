#include "leetcode.h"

using namespace std;
using namespace osrm;
using namespace osrm::timing;

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
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next->next;
        slow = slow->next;
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

    /*
        1. check whether there is a cycle in the list or not
        2. if there is, find the node where the cycle begins 
    */

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

    // PS: try to prove why fast and slow would rendezvous at the starting node
    fast = head;
    while (fast != slow) {
        fast = fast->next;
        slow = slow->next;
    }
    return fast;
}

void hasCycle_scaffold(std::string input1, int pos, bool expectedResult) {
    ListNode* l1 = stringToListNode(input1);
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
    bool ans = ss.hasCycle(l1);
    if (ans == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

bool detectCycle_scaffold(const vector<int>& vi, int pos) {
    ListNode* l1 = vectorToListNode(vi);
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
    ListNode* ans = ss.detectCycle(l1);
    return q == ans;
}

void detectCycle_scaffold(std::string input1, int pos, bool expectedResult) {
    auto vi = stringTo1DArray<int>(input1);
    if (detectCycle_scaffold(vi, pos)) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

void basic_test() {
    util::Log(logESSENTIAL) << "Running hasCycle tests:";
    TIMER_START(hasCycle);
    hasCycle_scaffold("[]", -1, false);
    hasCycle_scaffold("[1]", -1, false);
    hasCycle_scaffold("[1,2,3,4]", -1, false);
    hasCycle_scaffold("[1,2,3,4,5]", -1, false);
    hasCycle_scaffold("[1]", 0, true);
    hasCycle_scaffold("[1,2,3,4,5]", 0, true);
    hasCycle_scaffold("[1,2,3,4]", 2, true);
    hasCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, true);
    TIMER_STOP(hasCycle);
    util::Log(logESSENTIAL) << "hasCycle: " << TIMER_MSEC(hasCycle) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running detectCycle tests:";
    TIMER_START(detectCycle);
    detectCycle_scaffold("[]", -1, false);
    detectCycle_scaffold("[1]", -1, false);
    detectCycle_scaffold("[1]", 0, true);
    detectCycle_scaffold("[1,2,3,4,5]", 2, true);
    detectCycle_scaffold("[1,2,3,4]", 2, true);
    detectCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, true);
    TIMER_STOP(detectCycle);
    util::Log(logESSENTIAL) << "detectCycle: " << TIMER_MSEC(detectCycle) << " milliseconds.";
}

void batch_test_scaffold(int test_array_scale) {
    int pos = -1;
    vector<int> vi; vi.reserve(test_array_scale);
    for (int i=0; i<100; ++i) {
        vi.clear();
        int n = rand() % test_array_scale + 1;
        for (int j=0; j<n; ++j) {
            vi.push_back(rand());
        }

        pos = rand() % (10*n + 1);

        if(!detectCycle_scaffold(vi, pos)) {
            util::Log(logERROR) << "Case(test_array_scale<" << test_array_scale 
                << ">, array_size<" << n << ">, pos<" << pos << ">) failed";
        }
    }
}

int main(int argc, char* argv[]) {
    util::LogPolicy::GetInstance().Unmute();

    basic_test();

    int test_array_scale = 100;
    if (argc > 1) {
        test_array_scale = std::atoi(argv[1]);
        if (test_array_scale <= 0) {
            cout << "test_array_scale must be positive, default to 100 if unspecified" << endl;
            return -1;
        }
    }

    util::Log(logESSENTIAL) << "Running batch tests:";
    TIMER_START(detectCycle_batch_test);
    batch_test_scaffold(test_array_scale);
    TIMER_STOP(detectCycle_batch_test);
    util::Log(logESSENTIAL) << "batch tests using " << TIMER_MSEC(detectCycle_batch_test) << " milliseconds";
}