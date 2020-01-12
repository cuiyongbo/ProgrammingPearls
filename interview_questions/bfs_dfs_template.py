#!/usr/bin/env python

from collections import defaultdict, deque

class Graph(object):
    def __init__(self, nodeCount):
        self.G = defaultdict(list)
        self.V = nodeCount

    def addEdge(self, u, v):
        self.G[u].append(v)

    def neighbors(self, u):
        return self.G[u]

    def bfs(self, start, end):
        step = 0
        q = deque([start])
        visited = set()
        while len(q) != 0:
            n = len(q)
            for i in range(n):
                u = q.popleft()
                if u in visited: continue

                if end == u:
                    print("path({}, {}) found after {} steps".format(start, end, step))
                    return True

                visited.add(u)
                step += 1

                for v in self.neighbors(u):
                    if v not in visited:
                        q.append(v)

        print("no way found for path({}, {})".format(start, end))
        return False

    def dfs(self, start, end):
        def helper(u):
            visited.add(u)
            for v in self.neighbors(u):
                if v not in visited:
                    if end == v or helper(v):
                        path.append(v)
                        return True
            return False

        path = list()
        visited = set()
        if helper(start):
            path.append(start)
            print("found path:")
            path.reverse()
            print(path)
        else:
            print("no way found for path({}, {})".format(start, end))


if __name__ == '__main__':
    graph = Graph(5)
    graph.addEdge(0, 1)
    graph.addEdge(0, 2)
    graph.addEdge(1, 2)
    graph.addEdge(2, 0)
    graph.addEdge(2, 3)
    graph.addEdge(3, 3)

    graph.bfs(0, 3)
    graph.dfs(0, 3)
