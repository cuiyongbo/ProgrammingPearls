#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 917, 925, 986, 855 */

typedef vector<int> Interval;

class Solution 
{
public:
    string reverseOnlyLetters(string input);
    bool isLongPressedName(string name, string typed);
    vector<Interval> intervalIntersection(vector<Interval>& A, vector<Interval>& B);
    int numRescueBoats(vector<int>& people, int limit);
};

string Solution::reverseOnlyLetters(string input)
{
    /*
        Given a string S, return the “reversed” string where all characters 
        that are not a letter stay in the same place, and all letters reverse 
        their positions.

        Example 1:
        Input: "ab-cd"
        Output: "dc-ba"

        Example 2:
        Input: "a-bC-dEf-ghIj"
        Output: "j-Ih-gfE-dCba"

        Example 3:
        Input: "Test1ng-Leet=code-Q!"
        Output: "Qedo1ct-eeLg=ntse-T!"
    */

    auto isletter = [](char c) { return ('A' <= c && c <= 'Z') ||
                                    ('a' <= c && c <= 'z'); };
    int len = (int)input.size();
    int l = 0;
    int r = len - 1;
    while(l < r)
    {
        while(l < r && !isletter(input[l]))
            ++l;

        while(l < r && !isletter(input[r]))
            --r;

        if(l < r) std::swap(input[l++], input[r--]);
    }
    return input;
}

bool Solution::isLongPressedName(string name, string typed)
{
    /*
        Your friend is typing his name into a keyboard.  Sometimes, when typing a character c, 
        the key might get long pressed, and the character will be typed 1 or more times.

        You examine the typed characters of the keyboard.  
        Return True if it is possible that it was your friends name, 
        with some characters (possibly none) being long pressed.
        Note: the characters of name and typed are lowercase letters.
    */

    int len1 = (int)name.size();
    int len2 = (int)typed.size();

    int p1=0, p2=0;
    while(p1<len1 && p2<len2)
    {
        if(name[p1] != typed[p2]) return false;

        char c = name[p1];
        int count1 = 0;
        while(p1<len1 && name[p1] == c)
        {
            count1++; p1++;
        }

        int count2 = 0;
        while(p2<len2 && typed[p2] == c)
        {
            count2++; p2++;
        }

        if(count1 > count2) return false;
    }
    return p1 == len1 && p2 == len2;
}

vector<Interval> Solution::intervalIntersection(vector<Interval>& A, vector<Interval>& B)
{
    /*
        Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.
        Return the intersection of these two interval lists.

        (Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  
        The intersection of two closed intervals is a set of real numbers that is either empty, or can be 
        represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

        Example 1:
        Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
        Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
    */

    auto isEmpty = [](const Interval& s)
    {
        return s[0] > s[1];
    };

    auto intersection = [](const Interval& l, const Interval& r)
    {
        Interval s {0, -1};
        if(l[0] > r[1] || r[0]  > l[1])
        {
            return s;
        }
        else
        {
            s[0] = std::max(l[0], r[0]);
            s[1] = std::min(l[1], r[1]);
            return s;
        }
    };

    vector<vector<int>> ans;

    int len1 = (int)A.size();
    int len2 = (int)B.size();
    for(int p1=0, p2=0; p1<len1&&p2<len2;)
    {
        auto s = intersection(A[p1], B[p2]);
        if(!isEmpty(s)) ans.push_back(s);
        (A[p1][1] < B[p2][1]) ? ++p1 : ++p2;
    }
    return ans;
}

int Solution::numRescueBoats(vector<int>& people, int limit)
{
    /*
        The i-th person has weight people[i], and each boat can carry a maximum weight of limit.
        Each boat carries at most 2 people at the same time, provided the sum of the weight of 
        those people is at most limit.
        Return the minimum number of boats to carry every given person.  
        (It is guaranteed each person can be carried by a boat.)

        Example 1:
        Input: people = [1,2], limit = 3
        Output: 1
        Explanation: 1 boat (1, 2)

        Example 2:
        Input: people = [3,2,2,1], limit = 3
        Output: 3
        Explanation: 3 boats (1, 2), (2) and (3)

        Example 3:
        Input: people = [3,5,3,4], limit = 5
        Output: 4
        Explanation: 4 boats (3), (3), (4), (5)
    */

    std::sort(people.rbegin(), people.rend());
    
    int ans = 0;

    int l=0;
    int r = (int)people.size() - 1;
    while(l <= r)
    {
        if(l != r && people[l] + people[r] <= limit)
        {
            r--;
        }

        ans++; l++;
    }
    return ans;
}   

void reverseOnlyLetters_scaffold(string input, string expectedResult)
{
    Solution ss;
    string actual = ss.reverseOnlyLetters(input);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void isLongPressedName_scaffold(string input1, string input2, bool expectedResult)
{
    Solution ss;
    bool actual = ss.isLongPressedName(input1, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

void intervalIntersection_scaffold(string input1, string input2, string expectedResult)
{
    Solution ss;
    vector<vector<int>> A = stringTo2DArray<int>(input1);
    vector<vector<int>> B = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.intervalIntersection(A, B);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) util::Log(logERROR) << numberVectorToString(s); 
    }
}

void numRescueBoats_scaffold(string input1, int input2, int expectedResult)
{
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.numRescueBoats(A, input2);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: " << actual;
    }
}

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running reverseOnlyLetters tests: ";
    TIMER_START(reverseOnlyLetters);
    reverseOnlyLetters_scaffold("ab-cd", "dc-ba");
    reverseOnlyLetters_scaffold("a-bC-dEf-ghIj", "j-Ih-gfE-dCba");
    reverseOnlyLetters_scaffold("Test1ng-Leet=code-Q!", "Qedo1ct-eeLg=ntse-T!");
    TIMER_STOP(reverseOnlyLetters);
    util::Log(logESSENTIAL) << "reverseOnlyLetters using " << TIMER_MSEC(reverseOnlyLetters) << " milliseconds";

    util::Log(logESSENTIAL) << "Running isLongPressedName tests: ";
    TIMER_START(isLongPressedName);
    isLongPressedName_scaffold("cherry", "cherry", true);
    isLongPressedName_scaffold("leelee", "lleeelee", true);
    isLongPressedName_scaffold("leelee", "lleeeel", false);
    isLongPressedName_scaffold("saeed", "ssaaeed", true);
    isLongPressedName_scaffold("alex", "aaleex", true);
    isLongPressedName_scaffold("alex", "alexd", false);
    TIMER_STOP(isLongPressedName);
    util::Log(logESSENTIAL) << "isLongPressedName using " << TIMER_MSEC(isLongPressedName) << " milliseconds";

    util::Log(logESSENTIAL) << "Running intervalIntersection tests: ";
    TIMER_START(intervalIntersection);
    intervalIntersection_scaffold("[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[1,5],[8,12],[15,24],[25,26]]", 
                                    "[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]");
    intervalIntersection_scaffold("[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[0,2],[5,10],[13,23],[24,25]]", 
                                    "[[0,2],[5,10],[13,23],[24,25]]");
    TIMER_STOP(intervalIntersection);
    util::Log(logESSENTIAL) << "intervalIntersection using " << TIMER_MSEC(intervalIntersection) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numRescueBoats tests: ";
    TIMER_START(numRescueBoats);
    numRescueBoats_scaffold("[1,2]", 3, 1 );
    numRescueBoats_scaffold("[3,2,2,1]", 3, 3);
    numRescueBoats_scaffold("[3,5,3,4]", 5, 4);
    TIMER_STOP(numRescueBoats);
    util::Log(logESSENTIAL) << "numRescueBoats using " << TIMER_MSEC(numRescueBoats) << " milliseconds";

}
