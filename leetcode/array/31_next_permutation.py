class Solution(object):
    def nextPermutation(self, nums):
        N = len(nums)
        can = False
        idx = 0
        for i in range(N-1,0,-1):
            if nums[i]>nums[i-1]:
                can = True
                idx = i-1
                break
        if not can:
            for i in range(0,int(N/2)):
                nums[i],nums[N-1-i] = nums[N-1-i],nums[i]
        else:
            print('can')
            minidx = N-1
            for j in range(N-1,idx,-1):
                if nums[j]>nums[idx]:
                    nums[idx],nums[j] = nums[j],nums[idx]
                    start = idx+1
                    end = N-1
                    while start<end:
                        nums[start],nums[end] = nums[end],nums[start]
                        start += 1
                        end -= 1
                    break
            