#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 721, 399, 737 */

class Solution
{
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
    bool areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs);
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);
};

vector<double> Solution::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries)
{
    /*
        Equations are given in the format A / B = k, 
        where A and B are variables represented as strings, 
        and k is a real number (floating point number). 
        Given some queries, return the answers. If the answer does not exist, return -1.0.

        Example:
            Given a / b = 2.0, b / c = 3.0.
            queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
            return [6.0, 0.5, -1.0, 1.0, -1.0 ].

        The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , 
        where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.
    */

    map<string, vector<string>> graph;
    map<pair<string, string>, double> weightFunc;
    auto buildGraph = [&]()
    {
        int n = equations.size();
        for(int i=0; i<n; ++i)
        {
            graph[equations[i][0]].push_back(equations[i][1]);
            graph[equations[i][1]].push_back(equations[i][0]);
            
            weightFunc[{equations[i][0], equations[i][1]}] = values[i];
            weightFunc[{equations[i][1], equations[i][0]}] = 1 / values[i];
        }
    };

    function<bool(string, string, set<string>&, vector<string>&)> dfs = [&](string u, string dest, set<string>& visited, vector<string>& path)
    {
        visited.emplace(u);
        path.push_back(u);
        if(u == dest) return true;
        for(const auto& v: graph[u])
        {
            if(visited.count(v) == 0)
            {
                if(dfs(v, dest, visited, path))
                {
                    return true;
                }
            }
        }
        path.pop_back();
        return false;
    };

    auto calculatePathWeight = [&](vector<string>& path)
    {
        double w = 1;
        int n = path.size();
        for(int i=0; i<n-1; ++i)
        {
            w = w * weightFunc[{path[i], path[i+1]}];
        }
        return w;
    };

    buildGraph();

    vector<double> ans;
    for(const auto& q: queries)
    {
        string u = q[0], v = q[1];
        if(graph.count(u) == 0 || graph.count(v) == 0)
        {
            ans.push_back(-1.0);
            continue;
        }

        if(u == v)
        {
            ans.push_back(1.0);
            continue;
        }

        set<string> visited;
        vector<string> path;
        if(dfs(u, v, visited, path))
        {
            double w = calculatePathWeight(path);
            ans.push_back(w);
        }
    }
    return ans;
}

bool Solution::areSentencesSimilarTwo(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs)
{
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

    if(words1.size() != words2.size())
        return false;
    
    if(words1 == words2)
        return true;

    int wordId = 0;
    map<string, int> wordToIdMap;
    for(const auto& p: pairs)
    {
        if(wordToIdMap.count(p[0]) == 0)
        {
            wordToIdMap[p[0]] = ++wordId;
        }

        if(wordToIdMap.count(p[1]) == 0)
        {
            wordToIdMap[p[1]] = ++wordId;
        }
    }

    if(wordToIdMap.empty())
        return false;

    int wordCount = words1.size();
    int dictCount = wordToIdMap.size();
    DisjointSet dsu(dictCount);
    for(const auto& p: pairs)
    {
        dsu.unionFunc(wordToIdMap[p[0]], wordToIdMap[p[1]]);
    }

    set<int> grp1;
    for(const auto& s: words1)
    {
        if(wordToIdMap[s] == 0)
            return false;
        
        grp1.emplace(dsu.find(wordToIdMap[s]));
    }

    for(const auto& s: words2)
    {
        if(wordToIdMap[s] == 0)
            return false;
        
        int g = dsu.find(wordToIdMap[s]);
        if(grp1.count(g) == 0)
            return false;
    }

    return true;
}

vector<vector<string>> Solution::accountsMerge(vector<vector<string>>& accounts)
{
    /*
        Given a list accounts, each element accounts[i] is a list of strings, 
        where the first element accounts[i][0] is a name, and the rest of the 
        elements are emails representing emails of the account. ([name: emails])

        Now, we would like to merge these accounts. Two accounts definitely belong to the same person
        if there is some email that is common to both accounts. Note that even if two accounts have the same name, 
        they may belong to different people as people could have the same name. 
        A person can have any number of accounts initially, but all of their accounts definitely have the same name.

        After merging the accounts, return the accounts in the following format: 
        the first element of each account is the name, and the rest of the elements are emails in sorted order. 
        The accounts themselves can be returned in any order.
    */

    map<string, int> emailToGroup;
    int nodeCount = accounts.size();
    DisjointSet dsu(nodeCount);
    for(int i=0; i<nodeCount; ++i)
    {
        int n = accounts[i].size();
        for(int j=1; j<n; ++j)
        {
            if(emailToGroup[accounts[i][j]] != 0)
            {
                dsu.unionFunc(emailToGroup[accounts[i][j]], i+1);
                continue;
            }

            emailToGroup[accounts[i][j]] = i+1;
        }
    }

    map<int, vector<string>> groups;
    for(int i=0; i<nodeCount; ++i)
    {
        int g = dsu.find(i+1) - 1;
        groups[g].insert(groups[g].end(), accounts[i].begin()+1, accounts[i].end());
    }

    vector<vector<string>> ans;
    ans.reserve(nodeCount);
    for(auto& it: groups)
    {
        std::sort(it.second.begin(), it.second.end());
        it.second.erase(std::unique(it.second.begin(), it.second.end()), it.second.end());

        vector<string> item;
        item.push_back(accounts[it.first][0]);
        item.insert(item.end(), it.second.begin(), it.second.end());
        ans.emplace_back(item);
    }
    return ans;
}

void calcEquation_scaffold(string equations, string values, string queries, string expectedResult)
{
    Solution ss;
    vector<vector<string>> ve = to2DStringArray(equations);
    vector<double> dv = stringToDoubleVector(values);
    vector<vector<string>> vq = to2DStringArray(queries);
    auto expected = stringToDoubleVector(expectedResult);
    auto actual = ss.calcEquation(ve, dv, vq);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case( equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case( equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Expected: " << expectedResult << ", actual: " << numberVectorToString(actual);
    }
}

void areSentencesSimilarTwo_scaffold(string s1, string s2, string dict, bool expectedResult)
{
    Solution ss;
    vector<string> words1 = toStringArray(s1);
    vector<string> words2 = toStringArray(s2);
    vector<vector<string>> pairs = to2DStringArray(dict);
    bool actual = ss.areSentencesSimilarTwo(words1, words2, pairs);
    if(actual == expectedResult)
    {
        util::Log(logESSENTIAL) << "Case(" << s1 << ", " << s2 << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << s1 << ", " << s2 << ", expectedResult: " << expectedResult << ") failed";
    }
}

void accountsMerge_scaffold(string input, string expectedResult)
{
    Solution ss;
    vector<vector<string>> accounts = to2DStringArray(input);
    vector<vector<string>> expected = to2DStringArray(expectedResult);
    vector<vector<string>> actual = ss.accountsMerge(accounts);
    if(actual == expected)
    {
        util::Log(logESSENTIAL) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    }
    else
    {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "actual: ";
        for(const auto& a: actual)
        {
            for(const auto& i: a)
            {
                util::Log(logERROR) << i;
            }
        }
    }
}

int main()
{
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

}