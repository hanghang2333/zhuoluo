class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        n = len(nums1)+len(nums2)
        if n%2==0:
            before = self.findKth(nums1,nums2,int(n/2))
            after = self.findKth(nums1,nums2,int(n/2)+1)
            result = (before+after)*1.0/2
            return result
        else:
            result = self.findKth(nums1,nums2,int(n/2)+1)
            return result
    
    def findKth(self,A,B,k):
        if len(A)==0:
            return B[k-1]
        if len(B)==0:
             return A[k-1]
        if k==1:
            return min(A[0],B[0])
        a = A[int(k/2)-1] if len(A)>=int(k/2) else None
        b = B[int(k/2)-1] if len(B)>=int(k/2) else None
        if b is None or (a is not None and a<b):
            return self.findKth(A[int(k/2):],B,k-int(k/2))
        else:
            return self.findKth(A,B[int(k/2):],k-int(k/2))