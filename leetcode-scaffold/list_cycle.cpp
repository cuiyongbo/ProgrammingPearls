#include "leetcode.h"

using namespace std;
using namespace osrm;
using namespace osrm::timing;

/* leetcode: 141, 142 */

class Solution {
public:
    bool hasCycle(ListNode *head);
    ListNode *detectCycle(ListNode *head);
};

bool Solution::hasCycle(ListNode *head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast != nullptr) {
        fast = fast->next;
        if(fast == nullptr) {
            break;
        }
        fast = fast->next;    
        slow = slow->next;
        if (slow == fast) {
            break;
        }
    }
    return fast != nullptr;
}

ListNode* Solution::detectCycle(ListNode *head) {
    /* 
        Given a linked list, return the node where the cycle begins.
        If there is no cycle, return null.
    */

    ListNode* slow = head;
    ListNode* fast = head;
    while (fast != nullptr) {
        fast = fast->next;
        if(fast == nullptr) {
            break;
        }
        fast = fast->next;    
        slow = slow->next;
        if (slow == fast) {
            break;
        }
    }

    if (fast != nullptr) {
        fast = head;
        while (fast != slow) {
            slow = slow->next;
            fast = fast->next;
        }
    }
    return fast;
}

void hasCycle_scaffold(string input1, int pivot, bool expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* p = nullptr;
    ListNode* q = nullptr;
    ListNode* head = l1;
    while (head != nullptr) {
        p = head;
        if(head->val == pivot) {
            q = head;
        }
        head = head->next;
    }
    if(p != nullptr) {
        p->next = q;
    }

    Solution ss;
    bool ans = ss.hasCycle(l1);
    if(ans == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

void detectCycle_scaffold(string input1, int pivot, bool expectedResult) {
    ListNode* l1 = stringToListNode(input1);
    ListNode* p = nullptr;
    ListNode* q = nullptr;
    ListNode* head = l1;
    while (head != nullptr) {
        p = head;
        if(head->val == pivot) {
            q = head;
        }
        head = head->next;
    }
    if(p != nullptr) {
        p->next = q;
    }

    Solution ss;
    ListNode* ans = ss.detectCycle(l1);
    bool actual = ans != nullptr && ans->val == pivot;
    if(actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running hasCycle tests:";
    TIMER_START(hasCycle);
    hasCycle_scaffold("[]", 0, false);
    hasCycle_scaffold("[1]", 0, false);
    hasCycle_scaffold("[1]", 1, true);
    hasCycle_scaffold("[1,2,3,4,5]", 2, true);
    hasCycle_scaffold("[1,2,3,4]", 2, true);
    hasCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, true);
    TIMER_STOP(hasCycle);
    util::Log(logESSENTIAL) << "hasCycle: " << TIMER_MSEC(hasCycle) << " milliseconds.";

    util::Log(logESSENTIAL) << "Running detectCycle tests:";
    TIMER_START(detectCycle);
    detectCycle_scaffold("[]", 0, false);
    detectCycle_scaffold("[1]", 0, false);
    detectCycle_scaffold("[1]", 1, true);
    detectCycle_scaffold("[1,2,3,4,5]", 2, true);
    detectCycle_scaffold("[1,2,3,4]", 2, true);
    detectCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, true);
    TIMER_STOP(detectCycle);
    util::Log(logESSENTIAL) << "detectCycle: " << TIMER_MSEC(detectCycle) << " milliseconds.";
}
