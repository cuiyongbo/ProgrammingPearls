#include "leetcode.h"

using namespace std;
using namespace osrm;

/* leetcode: 323, 399, 721, 737, 839, 952, 924, 990 */

class Solution {
public:
    int countComponents(int n, vector<vector<int>>& edges);
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries);
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts);
    bool areSentencesSimilar_737(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs);
    int numSimilarGroups(vector<string>& A);
    int largestComponentSize(vector<int>& A);
    bool equationsPossible(vector<string>& equations);
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial);
};


/*
You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph.
Return the number of connected components in the graph.
*/
int Solution::countComponents(int n, vector<vector<int>>& edges) {

{ // dfs solution
    vector<vector<int>> graph(n);
    for (auto& e: edges) {
        graph[e[0]].push_back(e[1]);
        graph[e[1]].push_back(e[0]);
    }
    vector<int> visited(n, 0);
    function<void(int)> dfs = [&] (int u) {
        visited[u] = 1;
        for (auto v: graph[u]) {
            if (visited[v] == 0) {
                dfs(v);
            }
        }
        visited[u] = 2;
    };
    int ans = 0;
    for (int i=0; i<n; ++i) {
        if (visited[i] == 0) {
            dfs(i);
            ans++;
        }
    }
    return ans;
}

{ // disjoint set solution
    DisjointSet dsu(n);
    for (auto& e: edges) {
        dsu.unionFunc(e[0], e[1]);
    }
    set<int> groups;
    for (int i=0; i<n; i++) {
        groups.insert(dsu.find(i));
    }
    return groups.size();
}

}


/*
    Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). 
    Given some queries, return the answers. If the answer does not exist, return -1.0.
    Example:
        Given a / b = 2.0, b / c = 3.0.
        queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
        return [6.0, 0.5, -1.0, 1.0, -1.0].
    The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries,
    where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.
*/
vector<double> Solution::calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
    typedef std::pair<string, string> key_t;
    map<key_t, double> weight_map;
    map<string, vector<string>> graph;
    for (int i=0; i<equations.size(); ++i) {
        auto& p = equations[i];
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
        weight_map[{p[0], p[1]}] = values[i];
        weight_map[{p[1], p[0]}] = 1/values[i];
    }

    vector<key_t> path;
    set<string> visited;
    function<bool(string, string)> dfs = [&] (string u, string end) {
        if (u == end) {
            return true;
        }
        visited.insert(u);
        for (auto v: graph[u]) {
            if (visited.count(v) == 0) {
                path.emplace_back(u, v);
                if (dfs(v, end)) {
                    return true;
                }
                path.pop_back();
            }
        }
        return false;
    };
    auto eval_path = [&] (vector<key_t>& path) {
        double val = 1.0;
        for (auto& p: path) {
            val *= weight_map[p];
        }
        return val;
    };
    function<double(string, string)> calc_query = [&] (string start, string end) {
        if (graph.count(start)==0 || graph.count(end)==0) {
            return -1.0;
        }
        if (start == end) {
            return 1.0;
        }
        visited.clear();
        path.clear();
        if (dfs(start, end)) {
            return eval_path(path);
        } else {
            return -1.0;
        }
    };

    vector<double> ans;
    for (auto& p: queries) {
        ans.push_back(calc_query(p[0], p[1]));
    }
    return ans;
}


/*
    Given two sentences words1, words2 (each represented as an array of strings), and a list of similar word pairs pairs, determine if two sentences are similar.

    For example, words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar, 
    if the similar word pairs are pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]].

    Note that the similarity relation is transitive. For example, if “great” and “good” are similar, and “fine” and “good” are similar, then “great” and “fine” are similar.
    Similarity is also symmetric. For example, “great” and “fine” being similar is the same as “fine” and “great” being similar. (bidirectonal graph)

    Also, a word is always similar with itself. For example, the sentences words1 = ["great"], words2 = ["great"], 
    pairs = [] are similar, even though there are no specified similar word pairs.

    Two sentences are similar if:
        They have the same length (i.e., the same number of words)
        sentence1[i] and sentence2[i] are similar.
*/
bool Solution::areSentencesSimilar_737(vector<string>& words1, vector<string>& words2, vector<vector<string>>& pairs) {
    if (words1.size() != words2.size()) {
        return false;
    }
    map<string, vector<string>> graph;
    for (auto& p: pairs) {
        graph[p[0]].push_back(p[1]);
        graph[p[1]].push_back(p[0]);
    }
    set<string> visited;
    // return true if end is reachable from start
    function<bool(string,string)> dfs = [&] (string start, string end) {
        if (start == end) {
            return true;
        }
        visited.insert(start);
        for (auto& v: graph[start]) {
            if (visited.count(v) == 0) {
                if (dfs(v, end)) {
                    return true;
                }
            }
        }
        return false;
    };
    for (int i=0; i<words1.size(); ++i) {
        visited.clear();
        if (!dfs(words1[i], words2[i])) {
            return false;
        }
    }
    return true;
}


/*
    Given a list accounts, each element accounts[i] is a list of strings, 
    where the first element accounts[i][0] is a name, and the rest of 
    elements are emails representing emails of the account. ([name: email_list])

    Now, we would like to merge these accounts. Two accounts definitely belong to the same person
    if there is some email that is common to both accounts. Note that even if two accounts have the same name, 
    they may belong to different people as people could have the same name. 
    A person can have any number of accounts initially, but all of their accounts definitely have the same name.

    After merging the accounts, return the accounts in the following format: 
    the first element of each account is the name, and the rest of the elements are emails in sorted order. 
    The accounts themselves can be returned in any order.

    Hint: 
        solution one: use dfs to find connected components
        solution two: use disjoint_set to find connected components
*/
vector<vector<string>> Solution::accountsMerge(vector<vector<string>>& accounts) {

{ // disjoint set solution
    map<string, vector<int>> email_to_uid;
    for (int i=0; i<accounts.size(); ++i) {
        for (int j=1; j<accounts[i].size(); ++j) {
            email_to_uid[accounts[i][j]].push_back(i);
        }
    }
    int n = accounts.size();
    DisjointSet dsu(n);
    for (auto& p: email_to_uid) {
        for (int i=1; i<p.second.size(); ++i) {
            dsu.unionFunc(p.second[i-1], p.second[i]);
        }
    }
    map<int, vector<int>> user_groups;
    for (int i=0; i<n; ++i) {
        int g = dsu.find(i);
        user_groups[g].push_back(i);
    }
    vector<vector<string>> ans;
    for (auto& p: user_groups) {
        set<string> ss;
        vector<string> buffer;
        buffer.push_back(accounts[p.first][0]);
        for (auto i: p.second) {
            ss.insert(accounts[i].begin()+1, accounts[i].end());
        }
        buffer.insert(buffer.end(), ss.begin(), ss.end());
        ans.push_back(buffer);
    }
    return ans;
}

{ // dfs solution
    map<string,int> emails; // <email, account_id>
    map<string, vector<string>> graph;
    for (int i=0; i<accounts.size(); ++i) {
        int sz = accounts[i].size();
        for (int j=1; j<sz; ++j) {
            emails.emplace(accounts[i][j], i);
            graph[accounts[i][j]].push_back(accounts[i][j]); // in case for user(s) with only one email
            if (j < sz-1) {
                graph[accounts[i][j]].push_back(accounts[i][j+1]);
                graph[accounts[i][j+1]].push_back(accounts[i][j]);
            }
        }
    }

    set<string> visited;
    vector<string> path;
    std::function<void(string)> dfs = [&] (string u) {
        visited.insert(u);
        path.push_back(u);
        for (auto& v: graph[u]) {
            if (visited.count(v) == 0) {
                dfs(v);
            }
        }
    };

    vector<vector<string>> ans;
    for (auto& e: emails) {
        if (visited.count(e.first) == 0) {
            path.push_back(accounts[e.second][0]);
            dfs(e.first);
            std::sort(path.begin()+1, path.end());
            ans.push_back(path); 
            path.clear();
        }
    }
    return ans;
}

}


/*
    Two strings X and Y are similar if we can swap two letters (in different positions) of X, so that it equals Y.
    For example, "tars" and "rats" are similar (swapping at positions 0 and 2), and "rats" and "arts" are similar, but "star" is not similar to "tars", "rats", or "arts".

    Together, these form two connected groups by similarity: {"tars", "rats", "arts"} and {"star"}. Notice that "tars" and "arts" are in the same group even though they are not similar.  
    Formally, each group is such that a word is in the group if and only if it is similar to at least one other word in the group.

    We are given a list A of strings. Every string in A is an anagram of every other string in A. How many groups are there?

    Hint: use dfs/disjoint_set to find connected components
*/
int Solution::numSimilarGroups(vector<string>& A) {
    auto is_similar = [&] (string u, string v) {
        int diff = 0;
        for (int i=0; i<u.size(); ++i) {
            if (u[i] != v[i]) {
                diff++;
            }
        }
        return diff == 2;
    };

{ // disjoint_set solution
    int n = A.size();
    DisjointSet dsu(n);
    for (int i=0; i<n; ++i) {
        for (int j=i+1; j<n; ++j) {
            if (is_similar(A[i], A[j])) {
                dsu.unionFunc(i, j);
            }
        }
    }
    set<int> groups;
    for (int i=0; i<n; ++i) {
        groups.insert(dsu.find(i));
    }
    return groups.size();
}

{ // dfs solution
    int sz = A.size();
    map<string, vector<string>> graph;
    for (int i=0; i<sz; ++i) {
        graph[A[i]].push_back(A[i]); // in case for isolated nodes
        for (int j=i+1; j<sz; ++j) {
            if (is_similar(A[i], A[j])) {
                graph[A[i]].push_back(A[j]);
                graph[A[j]].push_back(A[i]);
            }
        }
    }
    set<string> visited;
    std::function<void(string)> dfs = [&] (string u) {
        visited.insert(u);
        for (auto& v: graph[u]) {
            if (visited.count(v) == 0) {
                dfs(v);
            }
        }
    };
    int ans = 0;
    for (auto& u: A) {
        if (visited.count(u) == 0) {
            dfs(u);
            ans++;
        }
    }
    return ans;
}

}


/*
    Given a non-empty array of unique positive integers A, consider the following graph:
    There are A.length nodes, labelled A[0] to A[A.length - 1]; There is an edge between A[i] and A[j] if and only if A[i] and A[j] share a common factor greater than 1.
    Return the size of the largest connected component in the graph.
    Constraints:
        All the values of nums are unique.
    Hint: use dfs/disjoint_set to find connected components, and return node count of the largest component
*/
int Solution::largestComponentSize(vector<int>& A) {

{ // disjoint_set solution
    int sz = A.size();
    DisjointSet dsu(sz);
    for (int i=0; i<sz; ++i) {
        for (int j=i+1; j<sz; ++j) {
            if (std::gcd(A[i], A[j]) > 1) { // require c++17
                dsu.unionFunc(i, j);
            }
        }
    }
    int ans = 0;
    map<int, int> groups; // group_id, group_size
    for (int i=0; i<sz; ++i) {
        int grp = dsu.find(i);
        groups[grp]++;
        ans = max(ans, groups[grp]);
    }
    return ans;
}

{ // dfs solution, Time Limit Execeeded
    auto is_similar = [&] (int u, int v) {
        int n = min(u, v);
        for (int i=2; i<=n; ++i) {
            if (u%i==0 && v%i==0) {
                return true;
            }
        }
        return false;
    };
    int sz = A.size();
    map<int, vector<int>> graph;
    for (int i=0; i<sz; ++i) {
        graph[A[i]].push_back(A[i]);
        for (int j=i+1; j<sz; ++j) {
            if (is_similar(A[i], A[j])) {
                graph[A[i]].push_back(A[j]);
                graph[A[j]].push_back(A[i]);
            }
        }
    }
    set<int> visited;
    std::function<int(int)> dfs = [&] (int u) {
        visited.insert(u);
        int node_count = 1;
        for (auto v: graph[u]) {
            if (visited.count(v) == 0) {
                node_count += dfs(v);
            }
        }
        return node_count;
    };
    int ans = 0;
    for (auto u: A) {
        if (visited.count(u) == 0) {
            ans = max(ans, dfs(u)); 
        }
    }
    return ans;
}

}


/*
    Given an array equations of strings that represent relationships between variables, 
    each string equations[i] has length 4 and takes one of two different forms: "a==b" or "a!=b".  
    Here, a and b are lowercase letters (not necessarily different) that represent one-letter variable names.
    Return true if and only if it is possible to assign integers to variable names so as to satisfy all the given equations.
    Constraints:
        equations[i].length == 4
        equations[i][0] is a lowercase letter.
        equations[i][1] is either '=' or '!'.
        equations[i][2] is '='.
        equations[i][3] is a lowercase letter.
*/
bool Solution::equationsPossible(vector<string>& equations) {
    DisjointSet dsu(128);
    for (auto& e: equations) {
        // put e[0] and e[3] into the same group
        if (e[1] == '=') {
            dsu.unionFunc(e[0], e[3]);
        }
    }
    for (auto& e: equations) {
        // e[0] and e[3] can not be in the same group
        if (e[1] == '!') {
            if (dsu.find(e[0]) == dsu.find(e[3])) {
                return false;
            }
        }
    }
    return true;
}


/*
    In a network of nodes, each node i is directly connected to another node j if and only if graph[i][j] = 1 (adjacency-matrix representation).

    Some nodes initial are initially infected by malware. When two nodes are directly connected 
    and at least one of those two nodes is infected by malware, both nodes will be infected by malware.  
    This spread of malware will continue until no more nodes can be infected in this manner.

    Suppose M(initial) is the final number of infected nodes after the spread of malware stops.

    We will remove one node from the initial list. Return the node that if removed, would minimize M(initial).
    If multiple nodes could be removed to minimize M(initial), return such a node with the smallest index. 

    Note that if a node was removed from the initial list of infected nodes, it may still be infected later as a result of the malware spread.      
*/
int Solution::minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
    int node_count = graph.size();
    DisjointSet dsu(node_count);
    for (int r=0; r<node_count; ++r) {
        for (int c=0; c<node_count; ++c) {
            if (graph[r][c] == 1) {
                dsu.unionFunc(r, c);
            }
        }
    }
    map<int, int> group_to_node_cnt_map; // component_id, node_cnt_in_the_component
    for (int i=0; i<node_count; ++i) {
        group_to_node_cnt_map[dsu.find(i)]++;
    }
    map<int, vector<int>> group_to_node_map; // component_id, node_idx_in_the_component_from_initial
    for (int i=0; i<initial.size(); ++i) {
        group_to_node_map[dsu.find(initial[i])].push_back(i);
    }
    int count = 0;
    int idx = INT32_MAX;
    for (auto& p: group_to_node_map) {
        if (p.second.size() == 1) {
            if (count < group_to_node_cnt_map[p.first]) {
                idx = p.second[0];
                count = group_to_node_cnt_map[p.first];
            } else if (count == group_to_node_cnt_map[p.first]) {
                idx = min(idx, p.second[0]);
            }
        }
    }
    return initial[idx == INT32_MAX ? 0 : idx];
}


void calcEquation_scaffold(string equations, string values, string queries, string expectedResult) {
    Solution ss;
    vector<vector<string>> ve = stringTo2DArray<string>(equations);
    vector<double> dv = stringTo1DArray<double>(values);
    vector<vector<string>> vq = stringTo2DArray<string>(queries);
    auto expected = stringTo1DArray<double>(expectedResult);
    auto actual = ss.calcEquation(ve, dv, vq);
    if (actual == expected) {
        util::Log(logINFO) << "Case(equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(equation: " << equations << ", values: " << values << ", queries: " << queries << ", expectedResult: " << expectedResult << ") failed, actual: " << numberVectorToString(actual);
    }
}


void areSentencesSimilarTwo_scaffold(string s1, string s2, string dict, bool expectedResult) {
    Solution ss;
    vector<string> words1 = stringTo1DArray<string>(s1);
    vector<string> words2 = stringTo1DArray<string>(s2);
    vector<vector<string>> pairs = stringTo2DArray<string>(dict);
    bool actual = ss.areSentencesSimilar_737(words1, words2, pairs);
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
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual == expected) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed";
        util::Log(logERROR) << "Actual: ";
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
        util::Log(logERROR) << "Case(" << input << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}


void equationsPossible_scaffold(string input, bool expectedResult) {
    Solution ss;
    vector<string> equations = stringTo1DArray<string>(input);
    bool actual = ss.equationsPossible(equations);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input << ", expected: " << expectedResult << ") failed, acutal: " << actual;
    }
}


void countComponents_scaffold(int n, string edges, int expectedResult) {
    Solution ss;
    vector<vector<int>> ve = stringTo2DArray<int>(edges);
    auto actual = ss.countComponents(n, ve);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << n << ", " << edges << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << n << ", " << edges << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}


void minMalwareSpread_scaffold(string input1, string input2, int expectedResult) {
    Solution ss;
    vector<vector<int>> graph = stringTo2DArray<int>(input1);
    vector<int> initial = stringTo1DArray<int>(input2);
    int actual = ss.minMalwareSpread(graph, initial);
    if (actual == expectedResult) {
        util::Log(logINFO) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") passed";
    } else {
        util::Log(logERROR) << "Case(" << input1 << ", " << input2 << ", expectedResult: " << expectedResult << ") failed, actual: " << actual;
    }
}


int main() {
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running calcEquation tests:";
    TIMER_START(calcEquation);
    calcEquation_scaffold("[[a,b], [b,c]]", "[2.0, 3.0]", "[[a,c], [b,a], [a,e], [a,a], [x,x]]", "[6.0, 0.5, -1, 1, -1]");
    TIMER_STOP(calcEquation);
    util::Log(logESSENTIAL) << "calcEquation using " << TIMER_MSEC(calcEquation) << " milliseconds";

    util::Log(logESSENTIAL) << "Running areSentencesSimilar_737 tests:";
    TIMER_START(areSentencesSimilar_737);
    areSentencesSimilarTwo_scaffold("[great]", "[great]", "[]", true);
    areSentencesSimilarTwo_scaffold("[great]", "[doubleplus, good]", "[[great, good]]", false);
    areSentencesSimilarTwo_scaffold("[great, acting, skill]", "[fine, drama, talent]", "[[great, good], [fine, good], [acting, drama], [skill, talent]]", true);
    TIMER_STOP(areSentencesSimilar_737);
    util::Log(logESSENTIAL) << "areSentencesSimilar_737 using " << TIMER_MSEC(areSentencesSimilar_737) << " milliseconds";

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
    numSimilarGroups_scaffold("[omv,ovm]", 1);
    TIMER_STOP(numSimilarGroups);
    util::Log(logESSENTIAL) << "numSimilarGroups using " << TIMER_MSEC(numSimilarGroups) << " milliseconds";

    util::Log(logESSENTIAL) << "Running largestComponentSize tests:";
    TIMER_START(largestComponentSize);
    largestComponentSize_scaffold("[4,6,15,35]", 4);
    largestComponentSize_scaffold("[20,50,9,63]", 2);
    largestComponentSize_scaffold("[2,3,6,7,4,12,21,39]", 8);
    largestComponentSize_scaffold("[4096,8195,14,24,4122,36,3761,6548,350,54,8249,4155,8252,70,9004,2120,4169,6224,4110,87,6233,4186,6238,4192,1382,4199,104,2153,1845,8310,4231,2185,4245,2212,4261,8359,2222,483,1506,8371,180,2230,6333,2238,2244,197,8391,6353,210,215,216,2265,4315,5872,222,224,8418,6371,4326,234,4331,4332,4334,6384,6387,4342,248,2298,4350,260,8454,2316,270,6419,277,6430,6431,8483,2341,310,311,8521,8525,4432,6483,6487,8541,8542,365,370,4473,396,8596,4501,8598,6551,408,2458,4510,4513,4515,6571,2476,8622,4530,8628,2486,8265,8632,8642,458,2512,4563,4569,4577,763,6629,6642,8700,512,4609,6659,5206,8961,6665,2571,4623,8721,4630,6679,8731,6685,8740,4650,557,8752,8884,4660,4665,4674,2629,4678,2632,4793,2635,6737,4690,2652,2656,2659,8807,6764,628,629,2679,6779,2685,2688,6786,4740,6795,2700,654,4751,5682,6807,678,5575,4785,2738,2739,2740,2743,1140,6844,4800,2754,9675,6853,712,4809,717,8925,8930,6883,4837,8936,8944,4849,754,3162,4853,6906,4859,766,769,812,2832,2834,796,799,8994,6947,4908,813,5619,817,6962,8329,4924,6975,2881,4930,9507,9034,9036,6989,9038,9041,7651,2904,9055,9069,7025,2931,2933,892,7038,7040,4994,2953,2955,7069,9119,5031,9129,5035,7086,1523,9143,9144,7099,5062,3018,9163,7121,9173,7130,3035,6309,3050,5102,3055,1010,1015,7160,7162,5115,5118,8704,7171,9220,9225,7179,7183,1040,7189,9240,9252,3112,7215,5171,5173,7226,9280,5186,7240,9295,5200,5201,5202,3158,9306,7268,1131,7276,5232,9332,1151,9345,1154,7303,7309,1182,7327,1184,1190,7337,7341,1199,9396,7353,9404,3275,7370,1227,7374,3284,3289,3299,5348,5349,1255,1266,5371,5376,9481,7438,8408,7444,7726,3355,1313,6363,3371,3373,5425,9527,9531,7486,3394,5443,5444,7494,9547,5454,9569,3429,5478,7533,3445,9591,4330,1406,1600,3468,3471,5527,7581,7591,9642,5557,9657,7071,7614,7618,3523,9668,1479,5576,7627,9676,9680,7639,9698,5603,1513,2642,9712,7763,5621,7081,9722,9728,9803,9733,3592,1546,5038,9753,1582,7730,3637,5696,9794,1609,951,7756,7758,5711,1617,9811,7764,1632,5738,9836,1647,3703,5754,9851,1661,5397,2668,9867,3728,9874,7827,1684,7834,1693,3747,3748,7856,9905,3766,1728,5828,2308,1735,7881,5834,7887,1745,6435,7896,1754,9950,8485,1760,9954,3822,3824,1777,1782,981,5891,7942,7946,8168,7958,5912,4834,5915,7973,5930,5941,5943,3899,5951,8005,8006,8161,1864,8013,3919,3926,8024,8038,6120,6004,6013,6015,6026,1933,6031,8081,3990,1944,1947,1952,6055,1963,8111,4024,6074,6075,5451,1997,2002,4053,8152,4057,4059,4063,2016,6113,4072,8179,4089]", 432);
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

    util::Log(logESSENTIAL) << "Running countComponents tests:";
    TIMER_START(countComponents);
    countComponents_scaffold(5, "[[0,1],[1,2],[3,4]]", 2);
    countComponents_scaffold(5, "[[0,1],[1,2],[2,3],[3,4]]", 1);
    TIMER_STOP(countComponents);
    util::Log(logESSENTIAL) << "countComponents using " << TIMER_MSEC(countComponents) << " milliseconds";

    util::Log(logESSENTIAL) << "Running minMalwareSpread tests:";
    TIMER_START(minMalwareSpread);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1]", 0);
    minMalwareSpread_scaffold("[[1,1,0],[1,1,0],[0,0,1]]", "[0,1,2]", 2);
    minMalwareSpread_scaffold("[[1,0,0],[0,1,0],[0,0,1]]", "[0,2]", 0);
    minMalwareSpread_scaffold("[[1,1,1],[1,1,1],[1,1,1]]", "[1,2]", 1);
    minMalwareSpread_scaffold("[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,1,0],[0,0,0,1,0,0],[0,0,1,0,1,0],[0,0,0,0,0,1]]", "[4,3]", 4);
    TIMER_STOP(minMalwareSpread);
    util::Log(logESSENTIAL) << "minMalwareSpread using " << TIMER_MSEC(minMalwareSpread) << " milliseconds";
}
