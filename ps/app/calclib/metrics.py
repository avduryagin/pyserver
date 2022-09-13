import numpy as np

class metrics:
    def __init__(self,w=None):
        self.w=w
    def euclidian(self,x=np.array([0,1]),y=np.array([1,0]),w=None):
        res = (x - y) ** 2
        if w is not None:
            self.w=w
        elif self.w is not None:
            return np.power(np.dot(res, self.w), 0.5)
        else:
            return np.power(np.sum(res), 0.5)
