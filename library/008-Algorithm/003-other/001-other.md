##N queen
```python
n = 0
   results = []
   cols = {}

   def attack(self, row, col):
       for c, r in self.cols.iteritems():
           if c - r == col - row or c + r == col + row:
               return True
       return False

   def search(self, row):
       if row == self.n:
           result = []
           for i in range(self.n):
               r = ['.'] * self.n
               r[self.cols[i]] = 'Q'
               result.append(''.join(r))
           self.results.append(result)
           return

       for col in range(self.n):
           if col in self.cols:
               continue
           if self.attack(row, col):
               continue
           self.cols[col] = row
           self.search(row + 1)
           del self.cols[col]

   def solveNQueens(self, n):
       self.n = n
       self.search(0)
       return self.results
```
##Maximum Product Subarray
```python
class Solution:
    # @param nums: an integer[]
    # @return: an integer
    def maxProduct(self, nums):
        # write your code here
        f, g = [], []
        f.append(nums[0])
        g.append(nums[0])
        for i in xrange(1, len(nums)):
            f.append(max(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
            g.append(min(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
        m = f[0]
        for i in xrange(1, len(f)): m = max(m, f[i])
        return m
```
##Dices Sum
扔 n 个骰子，向上面的数字之和为 S。给定 Given n，请列出所有可能的 S 值及其相应的概率。
```python
class Solution:
    # @param {int} n an integer
    # @return {tuple[]} a list of tuple(sum, probability)
    def dicesSum(self, n):
        # Write your code here
        results = []
        f = [[0 for j in xrange(6 * n + 1)] for i in xrange(n + 1)]

        for i in xrange(1, 7):
            f[1][i] = 1.0 / 6.0
        for i in xrange(2, n + 1):
            for j in xrange(i, 6 * n + 1):
                for k in xrange(1, 7):
                    if j > k:
                        f[i][j] += f[i - 1][j - k]
                f[i][j] /= 6.0

        for i in xrange(n, 6 * n + 1):
            results.append((i, f[n][i]))

        return results
```
