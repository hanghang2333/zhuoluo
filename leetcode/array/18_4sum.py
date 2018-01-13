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
    def fourSum(self, nums, target):
        nums = sorted(nums)
        result = set()
        for i in range(0,len(nums)-3):
            for j in range(i+1,len(nums)-2):
                nt = target - nums[i] - nums[j]
                res2 = self.twosum(nums[j+1:],nt)
                if len(res2)>0:
                    for res2s in res2:
                        result.add((nums[i],nums[j],res2s[0],res2s[1]))
        res = [list(i) for i in result]
        return res