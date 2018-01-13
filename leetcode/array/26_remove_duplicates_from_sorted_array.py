class Solution(object):
    def removeDuplicates(self, nums):
        if len(nums) == 0:
            return 0
        start = 0
        for i in range(1,len(nums)):
            if nums[i]==nums[start]:
                continue
            else:
                nums[start+1] = nums[i]
                start += 1
        return start+1