#include <iostream>
#include <vector>
#include <stack>

#define MAX_N 20001

using namespace std;

int n, m;
struct Node
{
    vector<int> adj;
    vector<int> rev_adj;
};

Node graph[MAX_N];

stack<int> S;
bool visited[MAX_N];

int component[MAX_N];
vector<int> components[MAX_N];
int numComponents;

void DFS1(int x)
{
    visited[x] = true;
    for (int i=0;i<graph[x].adj.size();i++)
    {
        if (!visited[graph[x].adj[i]]) DFS1(graph[x].adj[i]);
    }
    S.push(x);
}

void DFS2(int x)
{
    cout << x << " "; 
    component[x] = numComponents;
    components[numComponents].push_back(x);
    visited[x] = true;
    for (int i=0;i<graph[x].rev_adj.size();i++)
    {
        if (!visited[graph[x].rev_adj[i]]) DFS2(graph[x].rev_adj[i]);
    }
}

void Kosaraju()
{
    for (int i=0;i<n;i++)
    {
        if (!visited[i]) DFS1(i);
    }
    
    for (int i=0;i<n;i++)
    {
        visited[i] = false;
    }
    
    while (!S.empty())
    {
        int v = S.top(); S.pop();
        if (!visited[v])
        {
            printf("Component %d: ", numComponents);
            DFS2(v);
            numComponents++;
            printf("\n");
        }
    }
}

int main()
{
    vector<vector<int>> edges = {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 0},
        {1, 4},
        {4, 5},
        {4, 6},
        {6, 5}
    };

    n = 7;
    for(const auto& e: edges)
    {
        graph[e[0]].adj.push_back(e[1]);
        graph[e[1]].rev_adj.push_back(e[0]);
    }

    Kosaraju();
}