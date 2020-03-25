#include "leetcode.h"

using namespace std;
using namespace osrm;
using namespace osrm::timing;

/* leetcode: 21, 23, 147, 148 */

class Solution
{
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
    ListNode* mergeKLists(vector<ListNode*>& lists);
    ListNode* insertionSortList(ListNode* head);
    ListNode* mergeSortList(ListNode* head);

private:
    ListNode* mergeKLists_divideAndConquer(vector<ListNode*>& lists, int l, int r);
    ListNode* mergeKLists_minHeap(vector<ListNode*>& lists);
    ListNode* mergeSort(ListNode* head, ListNode* tail);
};

ListNode* Solution::mergeTwoLists(ListNode* l1, ListNode* l2)
{
    ListNode dummy;
    ListNode* p = &dummy;
    while(l1 != NULL && l2 != NULL)
    {
        if(l1->val > l2->val)
        {
            p->next = l2;
            p = p->next;
            l2 = l2->next;
        }
        else
        {
            p->next = l1;
            p = p->next;
            l1 = l1->next;
        }
    }
    p->next = (l1 != NULL) ? l1 : l2;
    return dummy.next;
}

ListNode* Solution::mergeKLists(vector<ListNode*>& lists)
{
    //return mergeKLists_divideAndConquer(lists, 0, lists.size()-1);
    return mergeKLists_minHeap(lists);
}

ListNode* Solution::mergeKLists_divideAndConquer(vector<ListNode*>& lists, int l, int r)
{
    if(l > r)
    {
        return NULL;
    }
    else if(l == r)
    {
        return lists[l];
    }
    else
    {
        int m = (r-l)/2 + l;
        ListNode* left = mergeKLists_divideAndConquer(lists, l, m);
        ListNode* right = mergeKLists_divideAndConquer(lists, m+1, r);
        return mergeTwoLists(left, right);
    }
}

ListNode* Solution::mergeKLists_minHeap(vector<ListNode*>& lists)
{
    auto cmp = [](ListNode* l, ListNode* r) { return l->val > r->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    for(auto l: lists)
    {
        while(l != NULL)
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

ListNode* Solution::insertionSortList(ListNode* head)
{
    if(head == NULL || head->next == NULL)
        return head;

    ListNode dummy;
    ListNode* sorted = &dummy;
    ListNode* tmp = head->next;
    head->next = sorted->next;
    sorted->next = head;
    head = tmp;

    while(head != NULL)
    {
        ListNode* p = &dummy;
        sorted = p->next;
        while(sorted != NULL && sorted->val < head->val)
        {
            p = sorted;
            sorted = sorted->next;
        }

        tmp = head->next;
        head->next = p->next;
        p->next = head;
        head = tmp;
    }
    return dummy.next;
}

ListNode* Solution::mergeSortList(ListNode* head)
{
    if(head == NULL || head->next == NULL)
        return head;

    return mergeSort(head, NULL);
}

ListNode* Solution::mergeSort(ListNode* head, ListNode* tail)
{
    if(head->next == tail)
    {
        // make sure that the trivial case returns an isolated node
        head->next = NULL;
        return head;
    }

    ListNode* p = head;
    ListNode* mid = head;
    while(p != tail)
    {
        p = p->next;
        if(p == tail) break;
        p = p->next;
        mid = mid->next;
    }

    ListNode* left = mergeSort(head, mid);
    ListNode* right = mergeSort(mid, tail);
    return mergeTwoLists(left, right);
}

void mergeTwoLists_scaffold(string input1, string input2, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l2 = stringToListNode(input2);
    ListNode* expected = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.mergeTwoLists(l1, l2);
    if (list_equal(ans, expected))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(ans);
    destroyLinkedList(expected);
}

void mergeKLists_scaffold(vector<string>& input,  string expectedResult)
{
    vector<ListNode*> lists;
    lists.reserve(input.size());
    for(auto& s: input)
    {
        lists.push_back(stringToListNode(s));
    }

    ListNode* expected = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.mergeKLists(lists);
    if(list_equal(ans, expected))
    {
        util::Log(logESSENTIAL) << "Case(" << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << expectedResult << ") failed";
    }

    destroyLinkedList(ans);
    destroyLinkedList(expected);
}

void insertionSortList_scaffold(string input1, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.insertionSortList(l1);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(ans);
    destroyLinkedList(l3);
}

void mergeSortList_scaffold(string input1, string expectedResult)
{
    ListNode* l1 = stringToListNode(input1);
    ListNode* l3 = stringToListNode(expectedResult);

    Solution ss;
    ListNode* ans = ss.mergeSortList(l1);
    if(list_equal(ans, l3))
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << expectedResult << ") failed";
    }

    destroyLinkedList(ans);
    destroyLinkedList(l3);
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    TIMER_START(mergeTwoLists);
    util::Log() << "Running mergeTwoLists tests:";
    mergeTwoLists_scaffold("[]", "[]", "[]");
    mergeTwoLists_scaffold("[1]", "[2,3]", "[1,2,3]");
    mergeTwoLists_scaffold("[1,2,3,4,5]", "[]", "[1,2,3,4,5]");
    mergeTwoLists_scaffold("[1,2,3,4,5,6,7,8,9,10]", "[5,6]", "[1,2,3,4,5,5,6,6,7,8,9,10]");
    TIMER_STOP(mergeTwoLists);
    util::Log() << "mergeTwoLists: " << TIMER_MSEC(mergeTwoLists) << " milliseconds.";


    TIMER_START(mergeKLists);
    util::Log() << "Running mergeKLists tests:";

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
    util::Log() << "mergeKLists: " << TIMER_MSEC(mergeKLists) << " milliseconds.";

    TIMER_START(insertionSortList);
    util::Log() << "Running insertionSortList tests:";
    insertionSortList_scaffold("[3,1,2]", "[1,2,3]");
    insertionSortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    insertionSortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    insertionSortList_scaffold("[2,2]", "[2,2]");
    insertionSortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(insertionSortList);
    util::Log() << "insertionSortList: " << TIMER_MSEC(insertionSortList) << " milliseconds.";

    TIMER_START(mergeSortList);
    util::Log() << "Running mergeSortList tests:";
    mergeSortList_scaffold("[3,1,2]", "[1,2,3]");
    mergeSortList_scaffold("[1,2,-3,4,0,5]", "[-3,0,1,2,4,5]");
    mergeSortList_scaffold("[10,9,8,7,6,5,4,3,2,1]", "[1,2,3,4,5,6,7,8,9,10]");
    mergeSortList_scaffold("[2,2]", "[2,2]");
    mergeSortList_scaffold("[1,2,3,4]", "[1,2,3,4]");
    TIMER_STOP(mergeSortList);
    util::Log() << "mergeSortList: " << TIMER_MSEC(mergeSortList) << " milliseconds.";
}
