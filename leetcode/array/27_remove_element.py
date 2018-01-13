class Solution(object):
    def removeElement(self, nums, val):
        start = 0
        for i in range(0,len(nums)):
            if nums[i]==val:
                continue
            else:
                nums[start]=nums[i]
                start += 1
        return start