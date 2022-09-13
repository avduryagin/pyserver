import numpy as np

def interseption(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    # print(A)
    # print(X)
    if mask1 & mask2:
        A[0] = x
        A[1] = y
        # print('returned ',A)
        return A
    if mask1:
        A[0] = x
        # print('returned ',A)
        return A
    if mask2:
        A[1] = y
        # print('returned ',A)
        return A
    if mask3:
        # print('returned ',A)
        return A.reshape(-1, shape)
    return np.array([],dtype=float)

def merge(A=np.array([]),B=np.array([]),shape=2):

    if (A[0] < B[1]) & (A[1] == B[0]):
        return np.array([A[0], B[1]])
    if (B[0] < A[1]) & (A[0] == B[1]):
        return np.array([B[0], A[1]])

    isp=interseption(A,B,shape=2)
    if isp.shape[0]>0:
        if A[0]<B[0]:
            return np.array([A[0],B[1]])
        else:
            return np.array([B[0], A[1]])

    else:
        return np.array([A,B])

def residual(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    if mask1 & mask2:
        #print('m12')
        A[1] = x
        if A.shape[0] == 3:
            B = np.array([y, b, A[2]])
        else:
            B = np.array([y, b],dtype=float)
        if (A[1]-A[0]>0)&(B[1]-B[0]>0):
            #print('both')
            return np.array([A, B])
        elif (A[1]-A[0]>0):
            #print('A')
            return np.array([A])
        elif (B[1]-B[0]>0):
            #print('B')
            return np.array([B])
        else:
            #print('empty')
            return np.array([],dtype=float)



    if mask1:
        A[1] = x
        #print(A)
        #print('mask1')
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)
    if mask2:
        A[0] = y
        #print('mask2')
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)
    if mask3:
        #print('mask3')
        return np.array([],dtype=float)
    return A.reshape(-1, shape)



def get_sets_residual(L, X, f=residual,shape=3):
    if shape==3:
        Y = np.array([], dtype=[('a', float), ('b', float), ('date', np.datetime64)]).reshape(-1, shape)
    else:
        Y = np.array([]).reshape(-1, shape)

    for l in L:
        y = f(l, X,shape=shape)

        if len(y) > 0:
            Y = np.vstack((Y, y))
    Y = np.vstack((Y, X.reshape(-1, shape)))
    return Y

def get_disjoint_sets(x=np.array([]),shape=2):
    if x.shape[0]>1:
        a=x[0]
        b=x[0:]
        x_=get_sets_residual(b,a,shape=shape)[:-1]
        y=get_disjoint_sets(x_,shape=shape)
        return np.vstack((a,y))
    else:
        return x

class linear_transform:
    def __init__(self, x=np.array([0,1]),y=np.array([0,1])):
        self.x1 = x[0]
        self.x2 = x[1]
        self.y1 = y[0]
        self.y2 = y[1]
        self.a2 = (self.y1 - self.y2) / (self.x1 - self.x2)
        self.a1 = (self.y1+self.y2 - self.a2*(self.x1+self.x2))*0.5

    def value(self, x):
        y=self.a1 + self.a2 * x
        return y


def mean_approach(*args,**kwargs):
    k=0
    s=0.
    for a in args:
        #if ~np.isnan(a):
        s+=a
        k+=1
    if k>0:
        s=s/k

    def value(x=0):
        return s

    return value

def cover(x=np.array([]).reshape(-1,2),mode='bw',length=100,size=100,c1=1,c0=0,cr=2, restrict=None):
    #c1-номер столбца, определяющего направление покрытия
    #c0 -номер столбца, который покрывается интервалами
    def split(bounds,x=np.array([]).reshape(-1,2),index=np.array([],dtype=np.int32),size=100,lbound=0,rbound=100,c1=1,c0=0,cr=2):
        if index.shape[0]==0:
            return
        i=index[0]
        cx=x[i,c0]
        y = restrict(i)
        #print(y)
        its=interseption(np.array([lbound,rbound]),y,shape=2).reshape(-1)
        #print(y,np.array([lbound,rbound]),its)
        a,b=get_interval(teta=size, current_point=cx,lbound=its[0], rbound=its[1], expand=False)
        lbounds=(lbound,a)
        rbounds=(b,rbound)
        lmask=x[index,c0]<a
        rmask=x[index,c0]>b
        lindex=index[lmask]
        rindex=index[rmask]
        bounds.append((indices[i],a,b))
        #print(np.array([i,a,b]),cx,llength,rlength)
        split(bounds,x,lindex,size=size,lbound=lbounds[0],rbound=lbounds[1],c1=c1,c0=c0)
        split(bounds,x,rindex, size=size, lbound=rbounds[0],rbound=rbounds[1], c1=c1, c0=c0)
    #mask=x[:,c0]<=length
    #x=x[mask]
    def get_bounds(x=np.array([]).reshape(-1,2),index=np.array([],dtype=np.int32),size=100,lbound=0,rbound=100):
        values=[]
        for i in index:
            try:
                cx = x[i, c0]
                #j = int(x[i, cr])
                y = restrict(i)
                its = interseption(np.array([lbound, rbound]), y, shape=2).reshape(-1)
                a, b = get_interval(teta=size, current_point=cx, lbound=its[0], rbound=its[1], expand=False)
                values.append((i,a,b))
            except IndexError: continue
        return np.array(values,dtype=[('i',np.int32),('a',np.float32),('b',np.float32)])
    def get(i=0):
        return re

    if restrict is None:
        re=np.array([0,length])
        restrict=get

    if (mode=='bw')|(mode=='fw'):
        sa=np.argsort(x[:,c1])
        indices = np.arange(x.shape[0])

        if mode=='bw':
            sa=np.flip(sa)
        bounds=[]
        split(bounds,x,index=sa,size=size,rbound=length,c1=c1,c0=c0)
        return np.array(bounds,dtype=[('i',np.int32),('a',np.float32),('b',np.float32)])
    elif mode=='reverse':
        index=np.arange(-x.shape[0],0)
    else:
        index=np.arange(x.shape[0])
    return get_bounds(x,index,size=size,rbound=length)

def get_interval(teta=100, k=1, current_point=0, lbound=0,rbound=100, expand=True, intervals=np.array([]).reshape(-1, 2)):
    # if current_point>lenght: return None
    teta = np.abs(teta)
    k = np.abs(k)
    a = current_point - k * teta
    b = current_point + k * teta
    if (a < lbound) & (b > rbound):
        a = lbound
        b = rbound
    if expand:
        if (a < lbound) & (b <= rbound):
            b = b - a
            a = lbound
            if b > rbound:
                b = rbound
        if (a >= lbound) & (b > rbound):
            a = a - (b - rbound)
            b = rbound
            if a < lbound:
                a = lbound
    else:
        if (a < lbound) & (b <= rbound):
            a = lbound
            b = b
        if (a >= lbound) & (b > rbound):
            a = a
            b = rbound
    # print(a,' ',b)
    if intervals.shape[0] > 0:
        for i in np.arange(intervals.shape[0]):
            x = intervals[i, 0]
            y = intervals[i, 1]

            mask1 = x <= a <= y
            mask2 = x <= b <= y
            if mask1 & mask2:
                a = current_point
                b = a
                return a, b
            if mask1:
                a = y
            if mask2:
                b = x

    return a, b

def get_unique_sets(x=np.array([],dtype=np.float32).reshape(-1,2)):
    mask=np.isnan(x[:,0])
    x=x[~mask]
    return np.unique(x,axis=0)


def masked(x=np.array([]),mask=np.array([],dtype=bool),val=0.):
    def get(index):
        try:
            if mask[index]:
                return val
            else:
                return x[index]
        except(IndexError):
            return val
    return get


