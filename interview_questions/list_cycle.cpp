#include "leetcode.h"

using namespace std;
using namespace osrm;

class Solution
{
public:
    bool hasCycle(ListNode *head);
};

bool Solution::hasCycle(ListNode *head)
{
    ListNode* slow = head;
    ListNode* fast = head;

    while(fast != NULL && fast->next != NULL)
    {
        fast = fast->next->next;
        slow = slow->next;

        if (slow == next)
            break;
    }
    return fast != NULL;
}

void hasCycle_scaffold(string input1, bool expectedResult)
{
    ListNode* l1 = stringToListNode(input1);

    Solution ss;
    bool ans = ss.swapPairs(l1);
    if(ans == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(l1);
}

int main()
{

    util::LogPolicy::GetInstance().Unmute();

    util::log() << "Running hasCycle tests:";
    hasCycle_scaffold("[]", false);
    hasCycle_scaffold("[1]", false);
}
