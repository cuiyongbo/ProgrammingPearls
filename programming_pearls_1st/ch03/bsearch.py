#!/usr/bin/python
#coding: utf-8

def bsearch_floor(key, nums):
	"""
	find the largest element in nums not greater than key
	"""
	if key < nums[0]:
		return (None, "key should be not less than the smallest element in nums")
	elif key >= nums[-1]:
		return len(nums)-1, nums[-1]

	start = 0
	end = len(nums) - 1
	while start <= end:
		mid = start + (end - start)/2
		if nums[mid] > key:
			end = mid - 1
		else:
			start = mid + 1
		
	return end, nums[end]


def bsearch_ceil(key, nums):
	"""
	find the smallest element in nums not less than key
	"""

	if key > nums[-1]:
		return (None,  "key should be not greater the largest element in nums")

	start = 0
	end = len(nums) - 1
	while start <= end:
		mid = start + (end - start)/2
		if nums[mid] < key:
			start = mid + 1
		else:
			end = mid - 1
		
	return start, nums[start]
	
	


if __name__ == "__main__":
	
	nums1 = [0, 1, 3,3,3, 5, 7, 9, 9]
	nums2 = range(1, 10, 2)
	nums3 = range(0, 10, 2)

	for array in [nums1, nums2, nums3]:
		print "test suit of", array
		for i in xrange(10):
#			print bsearch_floor(i, array)
			print bsearch_ceil(i, array)

