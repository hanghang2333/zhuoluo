class Solution(object):
    def twosum(self,nums,target):
        s = set()
        result = []
        for i in nums:
            if target-i in s:
                result.append([target-i,i])
            else:
                s.add(i)
        return result
    def threeSum(self, nums):
        nums = sorted(nums)
        result = set()
        for idx,i in enumerate(nums):
            if idx>0 and nums[idx]==nums[idx-1]:
                continue
            tmp = self.twosum(nums[idx+1:],0-i)
            if len(tmp)!=0:
                for tmpi in tmp:
                    result.add((i,tmpi[0],tmpi[1]))
        res = []
        for i in result:
            res.append(list(i))
        return res
