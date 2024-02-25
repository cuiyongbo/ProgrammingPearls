*****************
Leetcode Schedule
*****************

Summary
=======

#. stack: First In Last Out
#. queue: First In First Out
#. priority_queue: max_heap, min_heap
#. binary tree
    * traversals in dfs
        * preorder traversal: root --> left --> right
        * inorder traversal: left --> root --> right
        * postorder traversal: left --> right --> root
        * programming in recursive way using dfs
        * programming in iterative way with **stack**
    * traversal in bfs
        * level order traversal with **queue**
    * binary search tree: **left(x)->val <= x->val <= right(x)->val**
        * the traversal sequence is in no-decending order when performing inorder traveral
        * search, insertion, deletion

#. list
    * list traversal
    * reverse a list with **stack**
    * sort a list: insertion sort, merge sort
    * cycle detection: fast-slow pointer
    * programming tricks:
        * dummy node
        * push_back and push_front
        * **cut out node from original list**

#. trie tree
    * prefix search

#. binary search
    * boilerplates
        * binary search
        * lower_bound search
        * upper_bound search
    * problems: matrix search, kth element

#. backtrace
    * combination: order of element doesn't matter
    * permutation: order of element does matter
    * partition arrays, string
    * tricks:
        * prune useless branches
        * bfs: find the minimum length from src to dst
        * dfs: enumerate all the possible paths from src to dst

#. graph
    * dfs: graph coloring, cycle detection
    * bfs: find the shortest path from A to B
    * disjoint set: connected components
    * graph search algorithm:
        * bfs: bfs explores equally in all directions and the cost of every edge in the graph is the same
        * dijkstra algorithm: dijkstra algorithm prioritizes edges with lower costs when searching, and the the cost of edges in the graph is different
        * A star algorithm: A star is a modification of dijkstra algorithm that is optimized for a single destination, and it prioritizes edges which lead close to destionation using heuristics


Divide and conquer
==================

* 169 Majority Element
* 153 Find Minimum in Rotated Sorted Array    
* 154 Find Minimum in Rotated Sorted Array II  
* 912 Sort an Array 
* 315 Count of Smaller Numbers After Self


Backtrace
=========

* 17 Letter Combinations of a Phone Number
* 39 Combination Sum
* 40 Combination Sum II    
* 77 Combinations
* 78 Subsets   
* 90 Subsets II
* 46 Permutations    
* 47 Permutations II    
* 784 Letter Case Permutation    
* 943 Find the Shortest Superstring (Unsolved)
* 996 Number of Squareful Arrays    
* 20 Valid Parentheses    
* 22 Generate Parentheses
* 301 Remove Invalid Parentheses    
* 37 Sudoku Solver
* 51 N-Queens
* 52 N-Queens II
* 79 Word Search
* 212 Word Search II  


Graph
=====

* 133 Clone Graph
* 138 Copy List with Random Pointer 
* 200 Number of Islands
* 547 Friend Circles
* 695 Max Area of Island 
* 733 Flood Fill       
* 827 Making A Large Island 
* 1162 As Far from Land as Possible   
* 1020 Number of Enclaves        
* 841 Keys and Rooms
* 1202 Smallest String With Swaps    
* 207 Course Schedule
* 210 Course Schedule II    
* 802 Find Eventual Safe States   
* 399 Evaluate Division
* 839 Similar String Groups   
* 952 Largest Component Size by Common Factor   
* 990 Satisfiability of Equality Equations 
* 721 Accounts Merge    
* 785 Is Graph Bipartite   
* 886 Possible Bipartition 
* 1042 Flower Planting With No Adjacent     
* 997 Find the Town Judge
* 433 Minimum Genetic Mutation
* 815 Bus Routes
* 863 All Nodes Distance K in Binary Tree  
* 1129 Shortest Path with Alternating Colors
* 1263 Minimum Moves to Move a Box to Their Target Location  
* 684 Redundant Connection    
* 685 Redundant Connection II  
* 1319 Number of Operations to Make Network Connected  
* 743 Network Delay Time  
* 787 Cheapest Flights Within K Stops
* 882 Reachable Nodes In Subdivided Graph (Don't get the intention of it)
* 924 Minimize Malware Spread    
* 1334 Find the City With the Smallest Number of Neighbors at a Threshold Distance

Dynamic Programming
===================

* Longest Common Subsequence: 1143
* Longest Increasing Subsequence: 300, 673, 1048, 674, 128


Advanced topics
===============

* 146 LRU Cache
* 460 LFU Cache
* Monotonic stack exercises: 901, 907, 1019, 42
* stock problem: https://grandyang.com/leetcode/309/

Miscellaneous exercises
=======================

* 3 Longest substring without repeated characters
* Calculator: 150, 224, 227, 772


.. rubric:: Footnotes

.. [#] `花花酱 leetcode problem list <https://zxi.mytechroad.com/blog/leetcode-problem-categories/>`_
.. [#] `leetcode on github <https://github.com/doocs/leetcode.git>`_
.. [#] `https://grandyang.com/leetcode/42/`_

