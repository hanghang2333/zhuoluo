class Solution:
    """
    @param: nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        left = 0
        right = len(nums)-1
        for i in range(len(nums)-1,-1,-1):
            if nums[i]!=nums[left]:
                right = i
                break
        if nums[left]<nums[right]:
            return left
        while(left<right):
            print(left,right)
            mid = int((left+right)/2)
            if nums[mid]>=nums[0]:
                left=mid+1
            else:
                right = mid
        return left
    def search(self, A, target):
        if A==None or len(A)==0:
            return -1
        min0 = self.findMin(A)
        leftc = min0
        rightc = len(A)-leftc
        def ys(n):
            if n<rightc:
                return n+min0
            else:
                return n-rightc
            return 0
        left,right = -1,len(A)
        while left+1<right:
            mid = int((left+right)/2)
            mids = ys(mid)
            if A[mids]<target:
                left = mid
            else:
                right = mid
        right = ys(right)
        if A[right]!=target:
            return -1
        else:
            return right