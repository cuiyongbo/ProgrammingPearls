#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 855, 917, 925, 986 */

typedef vector<int> Interval;

class Solution {
public:
    int numRescueBoats(vector<int>& people, int limit);
    string reverseOnlyLetters(string input);
    bool isLongPressedName(string name, string typed);
    vector<Interval> intervalIntersection(vector<Interval>& A, vector<Interval>& B);
};

string Solution::reverseOnlyLetters(string input) {
/*
    Given a string S, return the “reversed” string where all characters that are not a letter stay in the same place, and all letters reverse their positions.
*/
    auto is_letter = [](char c) { return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z'); };
    int sz = input.size();
    for (int i=0, j=sz-1; i<j;) {
        if (is_letter(input[i]) && is_letter(input[j])) {
            swap(input[i], input[j]);
            ++i; --j;
        } else if (!is_letter(input[i])) {
            ++i;
        } else {
            --j;
        }
    }
    return input;
}

bool Solution::isLongPressedName(string name, string typed) {
/*
    Your friend is typing his name into a keyboard. Sometimes, when typing a character c, the key might get long pressed, and the character will be typed one or more times.
    You examine the typed characters of the keyboard. Return True if it is possible that it was your friends name, with some characters (possibly none) being long pressed. 
    Note: the characters of name and typed are lowercase letters.
*/
    int len1 = name.size();
    int len2 = typed.size();
    int i=0, j=0; 
    while (i<len1&&j<len2) {
        if (name[i] != typed[j]) {
            break;
        }
        int c = name[i];
        int p1 = i;
        while (i<len1 && name[i]==c) {
            ++i;
        }
        int p2 = j;
        while (j<len2 && typed[j]==c) {
            ++j;
        }
        if (i-p1 > j-p2) {
            break;
        }
    }
    return i==len1 && j==len2;
}

vector<Interval> Solution::intervalIntersection(vector<Interval>& A, vector<Interval>& B) {
/*
    Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order. Return the intersection of these two interval lists.

    (Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  
    The intersection of two closed intervals is a set of real numbers that is either empty, or can be 
    represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

    Example 1:
    Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
    Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
*/
    int len1 = A.size();
    int len2 = B.size();
    int i=0, j=0;
    vector<Interval> ans;
    while (i<len1 && j<len2) {
        if (A[i][0] > B[j][1]) {
            ++j;
        } else if (B[j][0] > A[i][1]) {
            ++i;
        } else { // intersection is not empty
            ans.push_back({max(A[i][0], B[j][0]), min(A[i][1], B[j][1])});
            if (A[i][1] < B[j][1]) {
                ++i;
            } else {
                ++j;
            }
        }
    }
    return ans;
}

int Solution::numRescueBoats(vector<int>& people, int limit) {
/*
    The i-th person has weight people[i], and each boat can carry a maximum weight of limit.
    Each boat carries at most 2 people at the same time, provided the sum of the weight of those people is at most limit.
    Return the minimum number of boats to carry every given person. (It is guaranteed each person can be carried by a boat.)

    Example 2:
    Input: people = [3,2,2,1], limit = 3
    Output: 3
    Explanation: 3 boats (1, 2), (2) and (3)

    Example 3:
    Input: people = [3,5,3,4], limit = 5
    Output: 4
    Explanation: 4 boats (3), (3), (4), (5)
*/
    sort(people.begin(), people.end(), std::greater<int>());
    int ans = 0;
    int l = 0;
    int r = people.size() - 1;
    while (l <= r) {
        if (l != r && people[l]+people[r]<=limit) {
            --r;
        }
        ++ans; ++l;
    }
    return ans;
} 

void reverseOnlyLetters_scaffold(string input, string expectedResult) {
    Solution ss;
    string actual = ss.reverseOnlyLetters(input);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void isLongPressedName_scaffold(string input1, string input2, bool expectedResult) {
    Solution ss;
    bool actual = ss.isLongPressedName(input1, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

void intervalIntersection_scaffold(string input1, string input2, string expectedResult) {
    Solution ss;
    vector<vector<int>> A = stringTo2DArray<int>(input1);
    vector<vector<int>> B = stringTo2DArray<int>(input2);
    vector<vector<int>> expected = stringTo2DArray<int>(expectedResult);
    vector<vector<int>> actual = ss.intervalIntersection(A, B);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
        for(const auto& s: actual) {
            util::Log(logERROR) << numberVectorToString(s);
        } 
    }
}

void numRescueBoats_scaffold(string input1, int input2, int expectedResult) {
    Solution ss;
    vector<int> A = stringTo1DArray<int>(input1);
    int actual = ss.numRescueBoats(A, input2);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}

int main() {
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
    numRescueBoats_scaffold("[3,1,3,1,3]", 4, 3);
    TIMER_STOP(numRescueBoats);
    util::Log(logESSENTIAL) << "numRescueBoats using " << TIMER_MSEC(numRescueBoats) << " milliseconds";

}
