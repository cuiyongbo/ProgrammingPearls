#!/usr/bin/env python
# coding=utf-8

def binary_search(items, v):
    low = 0
    high = len(items)-1
    while low <= high:
        mid = int((low+high)/2)
        if(items[mid] == v):
            return mid
        elif(items[mid] < v):
            low = mid + 1
        else:
            high = mid - 1
    return None


def binary_search_leftmost(items, v):
    """
        return the first item index equal to v, 
		if v is found in items. Or return the rank
		of v in items, that is, the number of elements
		which are less than v.
    """
    low = 0
    high = len(items)
    while low < high:
        mid = int((low+high)/2)
        if(items[mid] < v):
            low = mid + 1
        else:
            high = mid
    return low
#    if(low < len(items) and items[low] == v):
#        return low
#    else: 
#		print "%d not found." % v
#		return low

def binary_search_rightmost(items, v):
    """
        return the last item index equal to v, 
		if v is found in items. Or return the 
		number of element that are less than v. 
    """
    low = 0
    high = len(items)
    while low < high:
        mid = int((low+high)/2)
        if(items[mid] <= v):
            low = mid + 1
        else:
            high = mid
        #print(low, high)	
    return low - 1
#    if(low+1 < len(items) and items[low-1] == v):
#        return low-1
#    else: 
#        return None


items = [1,2,3,4,5,6,76]
print binary_search(items, 3)
print binary_search(items, 6)

items2 = [1,2,2,2,2,3,4,5,6]
print binary_search_leftmost(items2, 2) 
print binary_search_leftmost(items2, 0) 
print binary_search_leftmost(items2, 1) 
print binary_search_leftmost(items2, 7) 
print binary_search_rightmost(items2, 2) 
print binary_search_rightmost(items2, 0) 
print binary_search_rightmost(items2, 7) 

