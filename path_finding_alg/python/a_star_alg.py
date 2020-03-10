#!/usr/bin/env python

from collections import deque

class AStar:
    def movementCost(self, current, neighbor): pass
    def heuristic(self, start, goal): pass
    def neighbors(self, current): pass

    def reconstructPath(self, cameFrom, goal):
        node = goal
        path = deque()
        while node != None:
            path.appendLeft(node)
            node = cameFrom[node]
        return path

    def getLowest(self, openSet, fScore):
        lowest = float("inf")
        lowestNode = None
        for node in openSet:
            if fScore[node] < lowest:
                lowest = fScore[node]
                lowestNode = node
        return node

    def aStar(self, start, goal):
        cameFrom = {}
        openSet = set([start])
        closedSet = set()
        gScore = {}
        gScore[start] = 0
        fScore = {}
        fScore[start] = gScore[start] + self.heuristic(start, goal)
        while len(openSet) ! = 0:
            current = self.getLowest(openSet, fScore)
            if current == goal: 
                break
            openSet.remove(current)
            closedSet.add(current)
            for next in self.neighbors(current):
                newCost = gScore[current] + self.movementCost(current, next)
                if next in closedSet and newCost >= gScore[next]:
                    continue
                if next not in closedSet or newCost < gScore[next]:
                    cameFrom[next] = current
                    gScore[next] = newCost
                    fScore[next] = newCost + self.heuristic(next, goal)
                    if next not in openSet:
                        openSet.add(next)
        return cameFrom