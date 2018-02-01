## 7

## 81. Data Stream Media

Numbers keep coming, return the median of numbers at every time a new number added.

```python
from heapq import *
class Solution:
    """
    @param: nums: A list of integers
    @return: the median of numbers
    """
    def addnum(self,num,small,large):
        if len(small)==len(large):
            heappush(large,-heappushpop(small,-num))
        else:
            heappush(small,-heappushpop(large,num))
    def findmedian(self,small,large):
        if len(small) == len(large):
            return -1*small[0]
        else:
            return large[0]
    def medianII(self, nums):
        # write your code here
        res = []
        small = []
        large = []
        for i in nums:
            self.addnum(i,small,large)
            res.append(self.findmedian(small,large))
        return res

```

## 103. Linked List Cycle II

Given a linked list, return the node where the cycle begins.

If there is no cycle, return `null`.

```python
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        # write your code here
        root = head
        if root==None or root.next==None:
            return None
        fast = head
        slow = head
        while fast!=None and slow!=None:
            fast = fast.next
            if fast:
                fast = fast.next
            slow = slow.next
            if fast == slow:
                break
        if fast==None:
            return None
        while head!=slow:
            head = head.next
            slow = slow.next
        return head
```

## 362. Sliding Window Maximum

Given an array of n integer with duplicate number, and a moving window(size k), move the window at each iteration from the start of the array, find the maximum number inside the window at each moving.

```python
class Solution:
    """
    @param: nums: A list of integers
    @param: k: An integer
    @return: The maximum number inside the window at each moving
    """
    def maxSlidingWindow(self, nums, k):
        # write your code here
        ans = []
        queue = []
        for i, v in enumerate(nums):
            if queue and queue[0] <= i - k:
                queue = queue[1:]
            while queue and nums[queue[-1]] < v:
                queue.pop()
            queue.append(i)
            if i + 1 >= k:
                ans.append(nums[queue[0]])
        return ans
```

## 192. Wildcard Matching

Implement wildcard pattern matching with support for `'?'` and `'*'`.

- `'?'` Matches any single character.
- `'*'` Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

```python
class Solution:
    """
    @param: s: A string
    @param: p: A string includes "?" and "*"
    @return: is Match?
    """
    def isMatch(self, s, p):
        # write your code here
        n = len(s)
        m = len(p)
        dp = [[False for j in range(m+1)] for i in range(n+1)]
        dp[0][0] = True
        for j in range(1,m+1):
            if p[j-1]=='*':
                dp[0][j] = dp[0][j-1]
        for i in range(1,n+1):
            for j in range(1,m+1):
                if p[j-1]=='*':
                    dp[i][j] = dp[i-1][j-1] or dp[i-1][j] or dp[i][j-1]
                else:
                    if s[i-1]==p[j-1] or p[j-1]=='?':
                        dp[i][j] = dp[i-1][j-1]
        return dp[n][m]

```

## 7. Binary Tree Serialization

Design an algorithm and write code to serialize and deserialize a binary tree. Writing the tree to a file is called 'serialization' and reading back from the file to reconstruct the exact same binary tree is 'deserialization'.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """
    def serialize(self,root):
        if root==None:
            return []
        queue = [root]
        res = []
        while queue:
            queuebak = []
            tmpres = []
            for i in queue:
                if i == None:
                    tmpres.append('#')
                    queuebak.append(None)
                    queuebak.append(None)
                else:
                    tmpres.append(i.val)
                    queuebak.append(i.left)
                    queuebak.append(i.right)
            res.append(tmpres)
            if set(queuebak)==set([None]):
                break
            queue = queuebak
        return res
    def deserialize(self,data):
        if data==[]:
            return None
        root = TreeNode(data[0][0])
        queue = [root]
        for i in range(1,len(data)):
            nextl = data[i]
            print(nextl)
            queuebak = []
            for j in range(len(nextl)):
                if nextl[j]=='#':
                    queuebak.append(None)
                else:
                    queuebak.append(TreeNode(nextl[j]))
            print(queuebak)
            print(queue)
            for j in range(len(queue)):
                if queue[j] == None:
                    continue
                queue[j].left = queuebak[j*2]
                queue[j].right = queuebak[j*2+1]
            queue = queuebak
        return root

```

## 71. Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the *zigzag level order* traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param: root: A Tree
    @return: A list of lists of integer include the zigzag level order traversal of its nodes' values.
    """
    def zigzagLevelOrder(self, root):
        # write your code here
        if root==None:
            return []
        start = 0
        queue = [root]
        res = []
        while queue:
            queuebak = []
            tmpres = []
            for i in queue:
                tmpres.append(i.val)
                if i.left:
                    queuebak.append(i.left)
                if i.right:
                    queuebak.append(i.right)
            if start%2==0:
                res.append(tmpres)
            else:
                res.append(tmpres[::-1])
            start += 1
            queue = queuebak
        return res

```

## 50. Product of Array Exclude Itself

Given an integers array A.

Define B[i] = A[0] * ... * A[i-1] * A[i+1] * ... * A[n-1], calculate B **WITHOUT** divide operation.

```python
class Solution:
    """
    @param: nums: Given an integers array A
    @return: A long long array B and B[i]= A[0] * ... * A[i-1] * A[i+1] * ... * A[n-1]
    """
    def productExcludeItself(self, nums):
        # write your code here
        product1 = []
        product2 = []
        start1 = 1
        start2 = 1
        for idx,i in enumerate(nums):
            start1 = start1 * i
            product1.append(start1)
        for i in range(len(nums)-1,-1,-1):
            start2 = start2 * nums[i]
            product2.append(start2)
        res = []
        for i in range(len(nums)):
            p1 = 1
            p2 = 1
            if i>=1:
                p1 = product1[i-1]
            if i<=len(nums)-2:
                p2 = product2[len(nums)-i-2]
            res.append(p1*p2)
        return res


```

## 112. Remove Duplicates from Sorted List

Given a sorted linked list, delete all duplicates such that each element appear only *once*.

```python
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: head: head is the head of the linked list
    @return: head of linked list
    """
    def deleteDuplicates(self, head):
        # write your code here
        if head ==None:
            return head
        headbak = head
        nownode = headbak.val
        headbak = headbak.next
        prehead = head
        while headbak:
            if headbak.val != nownode:
                prehead.next = headbak
                prehead = prehead.next
                nownode = headbak.val
                headbak = headbak.next
            else:
                headbak = headbak.next
        prehead.next = None
        return head

```

## 8

## 88. Lowest Common Ancestor

Given the root and two nodes in a Binary Tree. Find the lowest common ancestor(LCA) of the two nodes.

The lowest common ancestor is the node with largest depth which is the ancestor of both nodes.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        if root is None:
            return None
        if root is A or root is B:
            return root
        left = self.lowestCommonAncestor(root.left,A,B)
        right = self.lowestCommonAncestor(root.right,A,B)
        if left is not None and right is not None:
            return root
        if left is not None:
            return left
        if right is not None:
            return right
        return None
```

## 6

## 97. Maximum Depth of Binary Tree

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def helper(self,root):
        if root==None:
            return 0
        return max(self.helper(root.left),self.helper(root.right))+1
    def maxDepth(self, root):
        # write your code here
        return self.helper(root)
```

## 5

## 4. Ugly Number II

Ugly number is a number that only have factors `2`, `3` and `5`.

Design an algorithm to find the *n*th ugly number. The first 10 ugly numbers are `1, 2, 3, 4, 5, 6, 8, 9, 10, 12...`

```python
class Solution:
    """
    @param: n: An integer
    @return: the nth prime number as description.
    """
    def nthUglyNumber(self, n):
        # write your code here
        if n==1:
            return 1
        s2,s3,s5=0,0,0
        res = [1]
        for i in range(1,n):
            s2t,s3t,s5t = res[s2]*2,res[s3]*3,res[s5]*5
            minn = min(s2t,s3t,s5t)
            if s2t==minn:
                s2+=1
            if s3t==minn:
                s3+=1
            if s5t==minn:
                s5+=1
            res.append(minn)
        return res[-1]
```

## 379. Reorder array to construct the minimum number

Construct minimum number by reordering a given non-negative integer array. Arrange them such that they form the minimum number.

```python
class Solution:
    """
    @param: nums: n non-negative integer array
    @return: A string
    """
    def minNumber(self, nums):
        # write your code here
        nums = [str(i) for i in nums]
        lenn = [len(i) for i in nums]
        maxlen = max(lenn)
        dict1 = {}
        for i in nums:
            ii = i*maxlen
            ii = ii[0:maxlen]
            dict1[i]=[ii]
        print(dict1)
        nums.sort(key=lambda x:dict1[x])
        #nums.sort(key=lambda x:x,reverse=True)
        res = ''.join(nums)
        i = 0
        for i in range(len(res)):
            if res[i]!='0':
                break
        return res[i:]
```

## 380. Intersection of Two Linked Lists

Write a program to find the node at which the intersection of two singly linked lists begins.

```python
"""
Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
"""


class Solution:
    """
    @param: headA: the first list
    @param: headB: the second list
    @return: a ListNode
    """
    def getIntersectionNode(self, headA, headB):
        # write your code here
        if headA==None or headB==None:
            return None
        lenA = 0
        lenB = 0
        heada = headA
        headb = headB
        while heada:
            lenA += 1
            heada = heada.next
        while headb:
            lenB += 1
            headb = headb.next
        longr,short=0,0
        if lenA>=lenB:
            longr,short = headA,headB
        else:
            longr,short = headB,headA
        for i in range(abs(lenA-lenB)):
            longr = longr.next
        while longr:
            if longr==short:
                return longr
            else:
                longr = longr.next
                short = short.next
        return None
```

## 381. Spiral Matrix II

Given an integer n, generate a square matrix filled with elements from 1 to n^2 in spiral order.

```python
class Solution:
    """
    @param: n: An integer
    @return: a square matrix
    """
    def bianlimatrix(self,row,col,n,idn,matrix):
        rowm = row-n
        colm = col-n
        for i in range(n,colm+1):
            matrix[n][i]=idn[0]
            idn[0]+=1
            #print(n,i)

        if rowm>n:
            for i in range(n+1,rowm+1):
                matrix[i][colm]=idn[0]
                idn[0]+=1
                #print(i,colm)
        if rowm>n and colm>n:
            for i in range(colm-1,n-1,-1):
                matrix[rowm][i]=idn[0]
                idn[0]+=1
                #print(rowm,i)
        #print('ddd')
        if rowm>n+1:
            for i in range(colm-1,n,-1):
                matrix[i][n]=idn[0]
                idn[0]+=1
                #print(i,n)
    def generateMatrix(self, n):
        if n==0:
            return []
        # write your code here
        matrix = [[0]*n for i in range(n)]
        start = 0
        idn = [1]
        while n>start*2:
            self.bianlimatrix(n-1,n-1,start,idn,matrix)
            start +=1
        return matrix

S = Solution()
print(S.generateMatrix(3))

```

## 532. Reverse Pairs

For an array A, if i < j, and A [i] > A [j], called (A [i], A [j]) is a reverse pair.
return total of reverse pairs in A.

```python
class Solution:
    """
    @param: A: an array
    @return: total of reverse pairs
    """
    def reversePairs(self, A):
        # write your code here
        if len(A)==0:
            return 0
        c=[0]
        self.merges(A,c)
        return c[0]
    def merges(self,A,c):
        if len(A)==1:
            return A
        else:
            mid = int(len(A)/2)
            leftA = self.merges(A[:mid],c)
            rightA = self.merges(A[mid:],c)
        return self.merge(leftA,rightA,c)
    def merge(self,leftA,rightA,c):
        res = []
        lstart = 0
        rstart = 0
        while lstart<len(leftA) and rstart<len(rightA):
            if leftA[lstart]>rightA[rstart]:
                res.append(rightA[rstart])
                c[0]+=len(leftA)-lstart
                rstart += 1
            else:
                res.append(leftA[lstart])
                lstart += 1
        res = res+leftA[lstart:]
        res = res+rightA[rstart:]
        return res
```

## 41. Maximum Subarray

```python
class Solution:
    """
    @param: nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        dp = [0 for i in range(len(nums)+1)]
        for i in range(len(nums)):
            dp[i+1]=max(dp[i]+nums[i],nums[i])
        return max(dp[1:])
```

## 46. Majority Number

Given an array of integers, the majority number is the number that occurs `more than half` of the size of the array. Find it.

```python
class Solution:
    """
    @param: nums: a list of integers
    @return: find a  majority number
    """
    def majorityNumber(self, nums):
        # write your code here
        if nums==None or len(nums)==0:
            return
        d = {}
        #d[nums[0]]=1
        key = nums[0]
        count = 1
        for i in range(1,len(nums)):
            now = nums[i]
            if count==0:
                key = now
            elif now==key:
                count+=1
            else:
                count -=1
        return key
```

## 374. Spiral Matrix

Given a matrix of *m* x *n* elements (*m* rows, *n* columns), return all elements of the matrix in spiral order.

```python
class Solution:
    """
    @param: matrix: a matrix of m x n elements
    @return: an integer list
    """
    def spiralOrder(self, matrix):
        # write your code here
        if matrix==[]:
            return []
        start = 0
        col = len(matrix[0])
        row = len(matrix)
        res = []
        while col>start*2 and row>start*2:
            self.helper(matrix,row,col,start,res)
            start +=1
        return res
    def helper(self,matrix,row,col,start,res):
        endx = col-1-start
        endy = row-1-start
        for i in range(start,endx+1):
            res.append(matrix[start][i])
        if start<endy:
            for i in range(start+1,endy+1):
                res.append(matrix[i][endx])
        if start<endx and start<endy:
            for i in range(endx-1,start-1,-1):
                res.append(matrix[endy][i])
        if start<endx and start<endy-1:
            for i in range(endy-1,start,-1):
                res.append(matrix[i][start])
```

## 378. Convert Binary Search Tree to Doubly Linked List

Convert a binary search tree to doubly linked list with in-order traversal.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
Definition of Doubly-ListNode
class DoublyListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = self.prev = next
"""


class Solution:
    """
    @param: root: The root of tree
    @return: the head of doubly list node
    """
    def bstToDoublyList1(self, root,tmp):
        # write your code here
        if root==None:
            return
        now = DoublyListNode(root.val)
        if root.left:
            #print(root.left.val)
            self.bstToDoublyList1(root.left,tmp)
        now.prev = tmp[0]
        #print(now.val)
        if tmp[0]:
            tmp[0].next = now
        tmp[0]=now
        if root.right:
            self.bstToDoublyList1(root.right,tmp)

    def bstToDoublyList(self,root):
        tmp = [None]
        self.bstToDoublyList1(root,tmp)
        tmp = tmp[0]
        while tmp!=None and tmp.prev!=None:
            tmp = tmp.prev
        return tmp
```

## 376. Binary Tree Path Sum

Given a binary tree, find all paths that sum of the nodes in the path equals to a given number `target`.

A valid path is from root node to any of the leaf nodes.

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def find(self,root,target,tmpres,res):
        if root==None:
            return
        if root.val==target and root.left==None and root.right==None:
            tmpres1=tmpres+[root.val]
            res.append(tmpres1)
            return
        else:
            self.find(root.left,target-root.val,tmpres[:]+[root.val],res)
            self.find(root.right,target-root.val,tmpres[:]+[root.val],res)
    def binaryTreePathSum(self, root, target):
        # write your code here
        if root==None:
            return []
        tmpres = []
        res = []
        self.find(root,target,tmpres,res)
        return res
```

## 3

## 140. Fast Power

```python
class Solution:
    """
    @param: a: A 32bit integer
    @param: b: A 32bit integer
    @param: n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        if n==0:
            return 1%b
        if n==1:
            return a%b
        if n%2==1:
            return ((a%b)*(self.fastPower(a,b,int(n/2))%b)**2)%b
        else:
            return ((self.fastPower(a,b,int(n/2))%b)**2)%b
```

## 371. Print Numbers by Recursion

Print numbers from 1 to the largest number with N digits by recursion.

```python
class Solution:
    """
    @param: n: An integer
    @return: An array storing 1 to the largest number with n digits
    """
    def numbersByRecursion(self, n):
        # write your code here
        res = []
        if n==0:
            return res
        else:
            self.helper(n,res)
        return res
    def helper(self,n,res):
        if n==0:
            return 1
        base = self.helper(n-1,res)
        size = len(res)
        for i in range(1,10):
            curbase = i*base
            res.append(curbase)
            for j in range(size):
                res.append(curbase+res[j])
        return base*10
```

## 35. Reverse Linked List

```python
"""
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param: head: n
    @return: The new head of reversed linked list.
    """
    def reverse(self, head):
        # write your code here
        if head == None or head.next==None:
            return head
        pre = None
        headnow = head
        while True:
            headnext = head.next
            head.next = pre
            pre = head
            if headnext==None:
                return head
            else:
                head = headnext
```

## 174. Remove Nth Node From End of List

```python
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param: head: The first node of linked list.
    @param: n: An integer
    @return: The head of linked list.
    """
    def removeNthFromEnd(self, head, n):
        # write your code here
        count = 0
        headbak = head
        while headbak!=None:
            count +=1
            headbak = headbak.next
        if count==n:
            return head.next
        ind = count-n#3
        headbak = head
        inds = 1
        while inds!=ind:
            inds+=1
            headbak=headbak.next
        headbak.next = headbak.next.next
        return head
```

## 245. Subtree

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param: T1: The roots of binary tree T1.
    @param: T2: The roots of binary tree T2.
    @return: True if T2 is a subtree of T1, or false.
    """
    def isequal(self,t1,t2):
        if t1==None or t2==None:
            return t1==t2
        if t1.val!=t2.val:
            return False
        return self.isequal(t1.left,t2.left) and self.isequal(t1.right,t2.right)

    def isSubtree(self, T1, T2):
        # write your code here
        if T2==None:
            return True
        if T1==None:
            return False
        if self.isequal(T1,T2):
            return True
        if self.isSubtree(T1.left,T2) or self.isSubtree(T1.right,T2):
            return True
        return False
```

## 2

## 73. Construct Binary Tree from Preorder and Inorder Traversal

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
class Solution:
    """
    @param preorder : A list of integers that preorder traversal of a tree
    @param inorder : A list of integers that inorder traversal of a tree
    @return : Root of a tree
    """
    def buildTree(self, preorder, inorder):
        # write your code here
        root = None
        if len(preorder)==0:
            return root
        else:
            root = TreeNode(preorder[0])
            ind = inorder.index(preorder[0])
            root.left = self.buildTree(preorder[1:1+ind],inorder[0:ind])
            root.right = self.buildTree(preorder[1+ind:],inorder[ind+1:])
            print(root.val)
        return root
```

## 159. Find Minimum in Rotated Sorted Array

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
            print(left,right)
            mid = int((left+right)/2)
            if nums[mid]>=nums[0]:
                left=mid+1
            else:
                right = mid
        return nums[left]
```
