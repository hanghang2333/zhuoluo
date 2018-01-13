class Solution(object):
    def twoSum(self, nums, target):
        s = {}
        for idx,i in enumerate(nums):
            if target-i in s:
                return [s[target-i],idx]
            else:
                s[i]=idx
        return [-1,-1]