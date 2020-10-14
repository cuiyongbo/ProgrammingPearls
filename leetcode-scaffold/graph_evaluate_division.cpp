#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 721, 399, 737, 839, 952, 990 */

class Solution {
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
    bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs);
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);
    int numSimilarGroups(vector<string>& A);
    int largestComponentSize(vector<int>& A);
    bool equationsPossible(vector<string>& equations);
};

vector<double> Solution::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    /*
        Equations are given in the format A / B = k, 
        where A and B are variables represented as strings, 
        and k is a real number (floating point number). 
        Given some queries, return the answers. 
        If the answer does not exist, return -1.0.

        Example:
            Given a / b = 2.0, b / c = 3.0.
            queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
            return [6.0, 0.5, -1.0, 1.0, -1.0 ].

        The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , 
        where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.
    */

    map<string, vector<string>> graph;
    map<pair<string, string>, double> weightFunc;
    int n = equations.size();
    for (int i=0; i<n; ++i) {
        graph[equations[i][0]].push_back(equations[i][1]);
        graph[equations[i][1]].push_back(equations[i][0]);
        weightFunc[{equations[i][0], equations[i][0]}] = 1;
        weightFunc[{equations[i][1], equations[i][1]}] = 1;
        weightFunc[{equations[i][0], equations[i][1]}] = values[i];
        weightFunc[{equations[i][1], equations[i][0]}] = 1 / values[i];
    }

    set<string> visited;
    vector<string> path;
    function<bool(string, string)> dfs = [&](string u, string dest) {
        if (graph.find(u) == graph.end() || graph.find(dest) == graph.end()) {
            return false;
        }
        visited.insert(u);
        if (u == dest) {
            return true;
        }
        for (const auto& v: graph[u]) {
            if (visited.count(v) == 0) {
                if (dfs(v, dest)) {
                    path.push_back(v);
                    return true;
                }
            }
        }
        return false;
    };

    auto calculatePathWeight = [&](vector<string>& path) {
        double w = 1;
        int n = path.size();
        for (int i=0; i<n-1; ++i) {
            w *= weightFunc[{path[i+1], path[i]}];
        }
        return w;
    };

    vector<double> ans;
    for (const auto& q: queries) {
        string u = q[0], v = q[1];
        if (dfs(u, v)) {
            path.push_back(u);
            double w = calculatePathWeight(path);
            ans.push_back(w);
        } else {
            ans.push_back(-1.0);
        }
        visited.clear();
        path.clear();
    }
    return ans;
}

bool Solution::areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs) {
    /*
        Given two sentences words1, words2 (each represented as an array of strings), 
        and a list of similar word pairs pairs, determine if two sentences are similar.

        For example, words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar, 
        if the similar word pairs are pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]].

        Note that the similarity relation is transitive. For example, if “great” and “good” are similar, 
        and “fine” and “good” are similar, then “great” and “fine” are similar.

        Similarity is also symmetric. For example, “great” and “fine” being similar is the same as “fine” and “great” being similar.

        Also, a word is always similar with itself. For example, the sentences words1 = ["great"], words2 = ["great"], 
        pairs = [] are similar, even though there are no specified similar word pairs.

        Finally, sentences can only be similar if they have the same number of words. 
        So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].
    */

    if (words1.size() != words2.size()) {
        return false;
    }
    if (words1 == words2) {
        return true;
    }

    int wordId = 0;
    map<string, int> wordIdMap;
    for (const auto& p: pairs) {
        if (wordIdMap.count(p[0]) == 0) {
            wordIdMap[p[0]] = ++wordId;
        }
        if (wordIdMap.count(p[1]) == 0) {
            wordIdMap[p[1]] = ++wordId;
        }
    }
    if (wordIdMap.empty()) {
        return false;
    }

    auto disjointset_solver = [&] () {
        int dictCount = wordIdMap.size();
        DisjointSet dsu(dictCount);
        for (const auto& p: pairs) {
            dsu.unionFunc(wordIdMap[p[0]], wordIdMap[p[1]]);
        }
        set<int> grp1;
        for (const auto& s: words1) {
            if(wordIdMap[s] == 0) {
                return false;
            }
            grp1.emplace(dsu.find(wordIdMap[s]));
        }
        for (const auto& s: words2) {
            if(wordIdMap[s] == 0) {
                return false;
            }
            int g = dsu.find(wordIdMap[s]);
            if(grp1.count(g) == 0) {
                return false;
            }
        }
        return true;
    };
    return disjointset_solver();

    map<int, vector<int>> dag;
    for (const auto& p : pairs) {
        dag[wordIdMap[p[0]]].push_back(wordIdMap[p[1]]);
        dag[wordIdMap[p[1]]].push_back(wordIdMap[p[0]]);
    }

    set<int> visited;
    function<bool(int, int)> dfs = [&] (int u, int dest) {
        if (dag.find(u) == dag.end() || dag.find(dest) == dag.end()) {
            return false;
        }
        visited.insert(u);
        if (u == dest) {
            return true;
        }
        for (auto v: dag[u]) {
            if (visited.count(v) == 0) {
                if (!dfs(v, dest)) {
                    return false;
                }
            }
        }
        return true;
    };

    // doesn't work if words1 and words2 are not in order
    int n = words1.size();
    for (int i=0; i<n; i++) {
        if (!dfs(wordIdMap[words1[i]], wordIdMap[words2[i]])) {
            return false;
        }
        visited.clear();
    }
    return true;
}

vector<vector<string>> Solution::accountsMerge(vector<vector<string>>& accounts) {
    /*
        Given a list accounts, each element accounts[i] is a list of strings, 
        where the first element accounts[i][0] is a name, and the rest of 
        elements are emails representing emails of the account. ([name: emails])

        Now, we would like to merge these accounts. Two accounts definitely belong to the same person
        if there is some email that is common to both accounts. Note that even if two accounts have the same name, 
        they may belong to different people as people could have the same name. 
        A person can have any number of accounts initially, but all of their accounts definitely have the same name.

        After merging the accounts, return the accounts in the following format: 
        the first element of each account is the name, and the rest of the elements are emails in sorted order. 
        The accounts themselves can be returned in any order.
    */

    int accountCount = accounts.size();
    map<string, vector<int>> emailToGrps;
    for (int i=0; i<accountCount; i++) {
        for (int j=1; j<accounts[i].size(); j++ ) {
            emailToGrps[accounts[i][j]].push_back(i);
        }
    }
    DisjointSet dsu(accountCount);
    for (const auto& p: emailToGrps) {
        for (int i=1; i<p.second.size(); i++) {
            dsu.unionFunc(p.second[i-1], p.second[i]);
        }
    }
    map<int, vector<int>> finalGrp;
    for (int i=0; i<accountCount; i++) {
        finalGrp[dsu.find(i)].push_back(i);
    }
    vector<string> row;
    set<string> emails;
    vector<vector<string>> ans;
    for (const auto& p: finalGrp) {
        row.clear();
        emails.clear();
        for (auto r: p.second) {
            for (int i=1; i<accounts[r].size(); i++) {
                emails.insert(accounts[r][i]);
            }
        }
        row.push_back(accounts[p.first][0]);
        row.insert(row.end(), emails.begin(), emails.end());
        ans.push_back(row);
    }
    return ans;
}

int Solution::numSimilarGroups(vector<string>& A) {
    /*
        Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.

        For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, 
        but "star" is not similar to "tars", "rats", or "arts".

        Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}.  
        Notice that "tars" and "arts" are in the same group even though they are not similar.  
        Formally, each group is such that a word is in the group if and only if it is similar 
        to at least one other word in the group.

        We are given a list A of strings. Every string in A is an anagram of every other string in A.  
        How many groups are there?
    */

    auto isSimilar = [&](int i, int j) {
        int diff = 0;
        for (int k=0; k<(int)A[i].size(); ++k) {
            if (A[i][k] != A[j][k]) {
                ++diff;
            }
        }
        return diff == 2;
    };

    int dictCount = A.size();
    DisjointSet dsu(dictCount);
    for (int i=0; i<dictCount; ++i) {
        for (int j=1; j<dictCount; ++j) {
            if(isSimilar(i, j)) {
                dsu.unionFunc(i, j);
            }
        }
    }
    set<int> groups;
    for (int i=0; i<dictCount; ++i) {
        groups.emplace(dsu.find(i));
    }
    return groups.size();
}

int Solution::largestComponentSize(vector<int>& A) {
    /*
        Given a non-empty array of unique positive integers A, consider the following graph:

        There are A.length nodes, labelled A[0] to A[A.length - 1];
        There is an edge between A[i] and A[j] if and only if A[i] and A[j] share a common factor greater than 1.

        Return the size of the largest connected component in the graph.
    */

    auto isSimilar = [&](int i, int j) {
        // require c++17
        // int cf = std::gcd(A[i], A[j]);
        // return  cf > 1;

        int min = std::min(A[i], A[j]);
        if ((A[i] % min == 0) && (A[j] % min == 0)) {
            return true;
        }
        for(int k=2; k<=min/2; ++k) {
            if ((A[i]%k==0) && (A[j]%k==0)) {
                return true;
            }
        }
        return false;
    };
    int dictCount = A.size();
    DisjointSet dsu(dictCount);
    for (int i=0; i<dictCount; ++i) {
        for (int j=1; j<dictCount; ++j) {
            if (isSimilar(i, j)) {
                dsu.unionFunc(i, j);
            }
        }
    }
    int ans = 0;
    map<int, int> groups;
    for(int i=0; i<dictCount; ++i) {
        groups[dsu.find(i)]++;
        ans = std::max(ans, groups[dsu.find(i)]);
    }
    return ans;
}

bool Solution::equationsPossible(vector<string>& equations) {
    /*
        Given an array equations of strings that represent relationships between variables, 
        each string equations[i] has length 4 and takes one of two different forms: "a==b" or "a!=b".  
        Here, a and b are lowercase letters (not necessarily different) that represent one-letter variable names.
        Return true if and only if it is possible to assign integers to variable names so as to satisfy all the given equations.    
    */

    DisjointSet eq_set(128);
    for (const auto& e: equations) {
        if (e[1] == '=') {
            eq_set.unionFunc(e[0], e[3]);
        }
    }
    for (const auto& e: equations) {
        // "==" is transitive but "!=" not 
        if (e[1] == '!' && eq_set.find(e[0]) == eq_set.find(e[3])) {
            return false;
        }
    }
    return true;
}


void calcEquation_scaffold(string equations, string values, string queries, string expectedResult) {
    Solution ss;
    vector<vector<string>> ve = stringTo2DArray<string>(equations);
    vector<double> dv = stringTo1DArray<double>(values);
    vector<vector<string>> vq = stringTo2DArray<string>(queries);
    auto expected = stringTo1DArray<double>(expectedResult);
    auto actual = ss.calcEquation(ve, dv, vq);
    if (actual == expected) {
        util::Log(logINFO) << "Case( equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case( equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", actual: " << numberVectorToString(actual);
    }
}

void areSentencesSimilarTwo_scaffold(string s1, string s2, string dict, bool expectedResult) {
    Solution ss;
    vector<string> words1 = stringTo1DArray<string>(s1);
    vector<string> words2 = stringTo1DArray<string>(s2);
    vector<vector<string>> pairs = stringTo2DArray<string>(dict);
    bool actual = ss.areSentencesSimilarTwo(words1, words2, pairs);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << s1 << ", " << s2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << s1 << ", " << s2 << ", expectedResult: " << expectedResult << ") failed";
    }
}

void accountsMerge_scaffold(string input, string expectedResult) {
    Solution ss;
    vector<vector<string>> accounts = stringTo2DArray<string>(input);
    vector<vector<string>> expected = stringTo2DArray<string>(expectedResult);
    vector<vector<string>> actual = ss.accountsMerge(accounts);
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "actual: ";
        for (const auto& a: actual) {
            for (const auto& i: a) {
                util::Log(logERROR) << i;
            }
        }
    }
}

void numSimilarGroups_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<string> accounts = stringTo1DArray<string>(input);
    int actual = ss.numSimilarGroups(accounts);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "actual: " << actual;
    }
}

void largestComponentSize_scaffold(string input, int expectedResult) {
    Solution ss;
    vector<int> graph = stringTo1DArray<int>(input);
    int actual = ss.largestComponentSize(graph);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "actual: " << actual;
    }
}

void equationsPossible_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<string> equations = stringTo1DArray<string>(input);
    bool actual = ss.equationsPossible(equations);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
    }
}

int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running calcEquation tests:";
    TIMER_START(calcEquation);
    calcEquation_scaffold("[[a,b], [b,c]]", "[2.0, 3.0]", "[[a,c], [b,a], [a,e], [a,a], [x,x]]", "[6.0, 0.5, -1, 1, -1]");
    TIMER_STOP(calcEquation);
    util::Log(logESSENTIAL) << "calcEquation using " << TIMER_MSEC(calcEquation) << " milliseconds";

    util::Log(logESSENTIAL) << "Running areSentencesSimilarTwo tests:";
    TIMER_START(areSentencesSimilarTwo);
    areSentencesSimilarTwo_scaffold("[great]", "[great]", "[]", true);
    areSentencesSimilarTwo_scaffold("[great]", "[doubleplus, good]", "[[great, good]]", false);
    areSentencesSimilarTwo_scaffold("[great, acting, skill]", "[fine, drama, talent]", "[[great, good], [fine, good], [acting, drama], [skill, talent]]", true);
    TIMER_STOP(areSentencesSimilarTwo);
    util::Log(logESSENTIAL) << "areSentencesSimilarTwo using " << TIMER_MSEC(areSentencesSimilarTwo) << " milliseconds";

    util::Log(logESSENTIAL) << "Running accountsMerge tests:";
    TIMER_START(accountsMerge);
    accountsMerge_scaffold("[[John, johnsmith@mail.com, john00@mail.com],"
                            "[John, johnnybravo@mail.com],"
                            "[John, johnsmith@mail.com, john_newyork@mail.com],"
                            "[Mary, mary@mail.com]]",
                            "[[John, johnnybravo@mail.com],"
                            "[John, john00@mail.com, john_newyork@mail.com, johnsmith@mail.com],"
                            "[Mary, mary@mail.com]]");
    TIMER_STOP(accountsMerge);
    util::Log(logESSENTIAL) << "accountsMerge using " << TIMER_MSEC(accountsMerge) << " milliseconds";

    util::Log(logESSENTIAL) << "Running numSimilarGroups tests:";
    TIMER_START(numSimilarGroups);
    numSimilarGroups_scaffold("[star, rats, arts, tars]", 2);
    TIMER_STOP(numSimilarGroups);
    util::Log(logESSENTIAL) << "numSimilarGroups using " << TIMER_MSEC(numSimilarGroups) << " milliseconds";

    util::Log(logESSENTIAL) << "Running largestComponentSize tests:";
    TIMER_START(largestComponentSize);
    largestComponentSize_scaffold("[4,6,15,35]", 4);
    largestComponentSize_scaffold("[20,50,9,63]", 2);
    largestComponentSize_scaffold("[2,3,6,7,4,12,21,39]", 8);
    TIMER_STOP(largestComponentSize);
    util::Log(logESSENTIAL) << "largestComponentSize using " << TIMER_MSEC(largestComponentSize) << " milliseconds";

    util::Log(logESSENTIAL) << "Running equationsPossible tests:";
    TIMER_START(equationsPossible);
    equationsPossible_scaffold("[a==b, b!=a]", false);
    equationsPossible_scaffold("[a==b, b==a]", true);
    equationsPossible_scaffold("[a==b, b==c, a==c]", true);
    equationsPossible_scaffold("[a==b, b!=c, a==c]", false);
    equationsPossible_scaffold("[a==b, c!=b, a==c]", false);
    equationsPossible_scaffold("[a==a, b==d, x!=z]", true);
    equationsPossible_scaffold("[a!=b, b!=c, c!=a]", true);
    TIMER_STOP(equationsPossible);
    util::Log(logESSENTIAL) << "equationsPossible using " << TIMER_MSEC(equationsPossible) << " milliseconds";
}
