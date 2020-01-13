#!/usr/bin/env python

# naive implementation
class DSU(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)

# DSU implementation of "Introduction to Algorithms, Chapter 21"
class OptDSU(object):
    def __init__(self, n):
        self.p = list(range(n))

        # self.rank[x] means the number of edges
        # in the longest simple path between x and
        # a descendant leaf
        self.r = list(range(n))

    # path compression with less recursion
    def find(self, u):
        while self.p[u] != u:
            self.p[u] = self.p[self.p[u]]
            u = self.p[u]
        return u

    # find-set with path compression
    def find_naive(self, u):
        while u != self.p[u]:
            self.p[u] = find_naive(self.p[u])
        return self.p[u]

    # union by rank
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)

        # no op since u and v are in the same set
        if pu == pv: return False

        if self.r[pu] < self.r[pv]:
            self.p[pu] = pv
        else:
            self.p[pv] = pu
            if self.r[pu] == self.r[pv]:
                self.r[pu] += 1
        return True

if __name__ == "__main__":

    M = []
    M.append([1,1,0])
    M.append([1,1,0])
    M.append([0,0,1])

    row, col = len(M), len(M[0])
    dsu = OptDSU(row)

    for r in range(row):
        for c in range(col):
            if M[r][c]:
                dsu.union(r, c)

    groups = set()
    for i in range(row):
        groups.add(dsu.find(i))
    print("Group Count: {}".format(len(groups)))
    print(groups)
