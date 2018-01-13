class Solution(object):
    def maxArea(self, height):
        start, end = 0,len(height)-1
        area = min(height[start],height[end])*(end-start)
        while start<end:
            if height[start]>height[end]:
                end = end - 1
            else:
                start = start + 1
            area = max(area, min(height[start],height[end])*(end-start))
        return area