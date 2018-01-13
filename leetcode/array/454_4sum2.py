class Solution(object):
    def fourSumCount(self, A, B, C, D):
        N = len(A)
        ab = {}
        for i in range(N):
            for j in range(N):
                if A[i]+B[j] in ab:
                    ab[A[i]+B[j]] += 1
                else:
                    ab[A[i]+B[j]] = 1 
        count = 0
        for i in range(N):
            for j in range(N):
                if 0-C[i]-D[j] in ab:
                    count += ab[0-C[i]-D[j]]
        return count