#!/usr/bin/env python

def combination(n, m, start, cur):
    if len(cur) == m:
        print(cur)
        return

    for i in range(start+1, n):
        cur.append(i+1)
        combination(n, m, i, cur)
        cur.pop()

def permutation(n, m, cur, used):
    if len(cur) == m:
        print(cur)
        return

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        cur.append(i+1)
        permutation(n, m, cur, used)
        cur.pop()
        used[i] = False


if __name__ == "__main__":
    n, m = 4, 2
    print("Combination({}, {}):".format(n, m))
    combination(n, m, -1, [])

    print("Permutation({}, {}):".format(n, m))
    used = [False for i in range(n)]
    permutation(n, m, [], used)
