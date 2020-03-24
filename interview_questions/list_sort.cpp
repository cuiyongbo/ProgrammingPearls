#include "leetcode.h"

using namespace std;
using namespace osrm;
using namespace osrm::timing;

/* leetcode: 23 */

class Solution
{
public:
    ListNode* mergeKLists(vector<ListNode*>& lists);    
};

void mergeKLists_scaffold(vector<string*>& input,  string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);

    ListNode* p = NULL;
    ListNode* q = NULL;
    ListNode* head = l1;
    while(head != NULL)
    {
        p = head;
        if(head->val == pivot)
        {
            q = head;
        }
        head = head->next;
    }

    if(p != NULL) p->next = q;

    Solution ss;
    ListNode* ans = ss.hasCycle(l1);
    if(list_equal(ans, expected))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    // destroyLinkedList(l1);
}


int main()
{
    util::LogPolicy::GetInstance().Unmute();

    TIMER_START(detectCycle);
    util::Log() << "Running detectCycle tests:";
    detectCycle_scaffold("[]", 0, false);
    detectCycle_scaffold("[1]", 0, false);
    detectCycle_scaffold("[1]", 1, true);
    detectCycle_scaffold("[1,2,3,4,5]", 2, true);
    detectCycle_scaffold("[1,2,3,4]", 2, true);
    detectCycle_scaffold("[1,2,3,4,5,6,7,8,9,10]", 5, true);
    TIMER_STOP(detectCycle);
    util::Log() << "detectCycle: " << TIMER_MSEC(detectCycle) << " milliseconds.";
}
