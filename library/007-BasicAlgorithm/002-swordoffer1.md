[TOC]
## Find Minimum in Rotated SOrted Array
### 题目:
Suppose a sorted array is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
Find the minimum element.
You may assume no duplicate exists in the array.
某个给定的不重复的有序数组进行了旋转操作，需要查找出这个数组里的最小元素。
Example:
Given [4, 5, 6, 7, 0, 1, 2] return 0
### 解答
旋转数组里最小元素查找。还有一种题目是旋转数组里查找某个元素坐标。思路一样不过会复杂一些。
旋转数组定义:
原数组[1,2,3,4,5,6]-->旋转两位[3,4,5,6,1,2].
查找肯定是二分查找。
特殊情况是旋转后和原数组一样，这种情况下第一个元素和最后一个关系肯定是小于。直接返回第一个即可。
一般情况就类似于上面举得例子，二分查找的思路就是每次通过mid元素的值进行判断而后减小一半的搜索长度。
初始化left=0，right=len-1.
而后计算得出mid。如果num[mid]的值大于等于num[0]，那么可以认为该mid是处于旋转数组的第一部分，这时候mid左边(包括mid)的都可以不要(因为第一部分的数组里一定不包含最小元素)。反之则是第二部分，这个时候右边的全部不要(第二部分mid右边的元素都比mid大)。
二分查找直到left和right相等。
完整代码如下:
```python
class Solution:
    """
    @param: nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        if nums[0]<nums[-1]:
            return nums[0]
        left = 0
        right = len(nums)-1
        while(left<right):
            mid = int((left+right)/2)
            if nums[mid]>=nums[0]:
                left=mid+1
            else:
                right = mid
        return nums[left]
```

那么有重复该如何做呢，举个例子就简单的，假设原数组是[1,2,5,5,5,6,6,7]旋转后如果是[2,5,5,5,6,6,7,1]则与前一种情况没有区别。只有当旋转操作的锚点恰好使得数组第一部分元素不再都是大于第二部分才会有问题，如[5,5,6,6,7,1,2,5]这样就会出错。那么解决方式也简单，处理这种情况就行了。直接对输入从后往前遍历，删除掉与首元素相同的即可。
完整代码如下:
```python
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
            return nums[left]
        while(left<right):
            print(left,right)
            mid = int((left+right)/2)
            if nums[mid]>=nums[0]:
                left=mid+1
            else:
                right = mid
        return nums[left]
```
## 二分查找相关
为了加深二分查找思路，不妨给出一个binarysearch的例子。如果查找到则返回位置，否则会分为两种情况，一种的lower bound即返回给定升序数组里大于等于目标值的最小索引，upper bound即返回给定升序数组里小于等于目标值的最大索引。举例来说:
在数组[1,2,3,4,6,7,8]中查找4则返回坐标3，查找5的话，按照lower bound返回6的坐标也就是4，按照upperbound的话返回4的坐标也就是3.实际应用里upperbound更常用，因为它表示了元素如果插入的话的插入位置。
下面分别给出两种情况对应的代码。

```python
    def lowerbound(nums,target):
        if nums==None or len(nums)==0:
            return -1
        left,right = -1,len(nums)
        while left+1<right:
            mid = int(left+right)/2
            if nums[mid]<target:
                left = mid
            else:
                right = mid
        return right
    def upperbound(self,nums,target):
        if nums==None or len(nums)==0:
            return -1
        left,right = -1,len(nums)
        while left+1<right:
            mid = int((left+right)/2)
            if (nums[mid]>target):
                right = mid
            else:
                left = mid
        return left
```
其中lower bound里初始化时left和right都在数组坐标范围之外,这样的好处是如果数组所有元素都比目标元素大，则left最后一直为-1，最后返回0符合要求。也可以理解为-1位置有一个负无穷的值。如果数组所有元素都大于目标元素，则最后返回的的是right的值。
针对要求的是lowerbound还是upperbound，程序稍微不同，对比着看一下。lower时要求的是大于等于的最小索引所以返回的是right，求大于等于也就是如果当前值大于等于则right可以直接跑到这，而小于的话则left可以跑到这。upper求的是小于等于的最大索引所以返回的是left。易知对于一个数组查找target，lowerbound和upperbound得到的结果永远都是相邻的。之所以还要区分，是因为存在查找到的情况，这种情况下我们需要返回查找到的值的索引而不是lowerbound或者upperbound。
二分查找思路简单但是写程序也是比较容易出bug。上面两个程序可以记住。
## 旋转数组里查找元素
有了旋转数组查找最小值和二分查找的基础，这个问题就很容易解答了。
对输入首先查找到最小值所在位置，而后可以写个函数作为真实索引和虚拟索引(也就是我们假设按照这个最小值将原数组重新整合好)的映射，而后就可以使用二分查找来完成这个程序了。(这个不是最优解，但复杂度也就log，并且似乎更容易理解也不容易出错)。并且上面也已经讨论了旋转数组里包含重复元素或者二分查找包含重复元素的情况，所以这里我们这里已经可以处理包含重复元素的情况了。
完整代码如下:
```python
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
```
