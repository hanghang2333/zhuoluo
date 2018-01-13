class Solution(object):
    def twosumclosest(self,nums,target):
        start,end = 0,len(nums)-1
        closest = abs(nums[0]+nums[-1]-target)
        closestp = nums[0]+nums[-1]
        while start<end:
            if nums[start]+nums[end]<target:
                start += 1
            else:
                end -= 1
            if start==end:
                continue
            tmpcp = nums[start]+nums[end]
            tmpc = abs(tmpcp-target)
            if tmpc<closest:
                closest,closestp = tmpc,tmpcp
        return closest,closestp

    def threeSumClosest(self, nums, target):
        nums = sorted(nums)
        closest = abs(sum(nums[0:3])-target)
        closestp = sum(nums[0:3])
        for idx,i in enumerate(nums):
            if idx>=len(nums)-2:
                continue
            nt = target-i
            tmpc,tmpcp = self.twosumclosest(nums[idx+1:],nt)
            if tmpc<closest:
                closest,closestp = tmpc,tmpcp+i
        return closestp
