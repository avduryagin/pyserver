import numpy as np
import os
#import pandas as pd
#import joblib
import pickle

class ClRe:
    def __init__(self,c=np.array([],dtype=float),r=np.array([],dtype=float),
                 t=np.array([],dtype=float),s=np.array([],dtype=float),shape=np.array([],dtype=int)):
        self.c=c
        self.r=r
        self.t=t
        self.s=s
        self.shape=shape
        self.indices=np.arange(c.shape[0])

    def get_items(self,mask=np.array([],dtype=bool),indices=np.array([],dtype=int)):
        if mask.shape[0]>0:
            return ClRe(self.c[mask],self.r[mask],self.t[mask],self.s[mask],self.shape[mask])
        elif indices.shape[0]>0:
            #print(indices)
            return ClRe(self.c[indices],self.r[indices],self.t[indices],self.s[mask],self.shape[indices])
        else:
            return None
class Generator:
    def __init__(self, classifier=None, regressor=None, col=None,path=None,modelfolder='models',regmodel='rfreg.sav',clmodel='rfc.sav',colfile='col.npy',rscale=1):
        #print('path ',os.path.dirname(os.path.abspath(__file__)))

        if path is None:
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)),modelfolder)
            #path=os.path.join(os.getcwd(),modelfolder)
        if regressor is not None:
            self.regressor=regressor
        else:
            self.regressor=pickle.load(open(os.path.join(path,regmodel), 'rb'))
        if classifier is not None:
            self.classifier=classifier
        else:
            self.classifier = pickle.load(open(os.path.join(path,clmodel), 'rb'))

        if col is not None:
            self.col=col
            ##print('took the dict')
        else:
            self.col = np.load(os.path.join(path,colfile),allow_pickle=True)[()]

        self.x = ClRe(c=np.array([]), r=np.array([]))
        self.rscale=rscale
        self.gindices = np.array([], dtype=int)
        self.mask = np.array([], dtype=bool)
        self.top = np.array([], dtype=float)
        self.prev = np.array([], dtype=float)
        self.columns = np.array([], dtype=int)
        self.down_stairs = dict({'ads05': 0.5, 'ads1': 1., 'ads2': 2., 'ads3': 3.})
        self.r=np.array([])
        self.p=np.array([])

    def get_next(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                              t=np.array([], dtype=float), s=np.array([], dtype=float), shape=np.array([], dtype=int)),
                 top=np.array([], dtype=float)):
        # прогнозирование класссификационной задачи
        prob = self.classifier.predict_proba(x.c)
        pred_mask=np.where(prob[:,1]>0.5)[0]
        #pred_mask = np.array(np.argmax(prob, axis=1), bool)
        #if pred_mask[pred_mask == True].shape[0] == 0:
        if pred_mask.shape[0]==0:
            return None, pred_mask,prob
        # для  1 прогнозируется следующая точка y
        y = self.regressor.predict(x.r[pred_mask]/self.rscale).reshape(-1)
        y=y*self.rscale
        prev = x.r[pred_mask][:, -1]
        delta = np.abs(y - prev)
        y = prev + delta
        emask = y == prev
        y[emask] = top[pred_mask][emask]
        y_hat = y * x.s[pred_mask]
        x_hat = x.get_items(mask=pred_mask)
        r_tilde = np.hstack((x_hat.r[:, 1:], y.reshape(-1, 1)))
        x_tilde, t_tilde, shape_tilde = self.get_new(x=x_hat.c, tau=y_hat, t=x_hat.t, shape=x_hat.shape)
        return ClRe(c=x_tilde, r=r_tilde, t=t_tilde, shape=shape_tilde, s=x.s[pred_mask]), pred_mask, prob[:, 1]


    def predict(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                                t=np.array([], dtype=float), s=np.array([], dtype=float),
                                shape=np.array([], dtype=int)),
                   top=np.array([], dtype=float), stop=10):

        self.x = x
        if self.x.c.shape[0]==0:
            self.proba=np.array([],dtype=np.float32)
            self.p = np.array([], dtype=np.float32)
            self.r = np.array([], dtype=np.float32)
            return self.proba
        self.top = top
        self.gindices = self.x.indices
        self.mask = np.ones(self.x.indices.shape[0], dtype=bool)
        self.proba = np.zeros(self.x.indices.shape[0], dtype=float)
        self.p0 = np.zeros(self.x.indices.shape[0], dtype=float)
        self.dp = np.zeros(self.x.indices.shape[0], dtype=float)
        self.indices = self.gindices
        r=[]
        p=[]
        cr=self.x.r[:,-1]
        cp=np.zeros(self.x.c.shape[0],dtype=np.float32)

        i = 1
        while (i < stop) & (self.x.indices.shape[0] > 0):
            cr_=cr.copy()
            cp_=np.zeros(cp.shape[0], dtype=float)

            y, pred_mask, probab = self.get_next(x=self.x, top=self.top)
            if y is None:
                r.append(cr)
                p.append(cp)
                self.p = np.array(p, dtype=np.float32)
                self.r = np.array(r, dtype=np.float32)
                return self.proba

            self.mask = (y.r[:, -1] <= self.top[pred_mask])
            pred_mask = pred_mask[self.mask]
            index = self.indices[pred_mask]

            if i == 1:
                self.p0 = probab[pred_mask]
                self.dt = probab[pred_mask]
                self.proba[index]=probab[pred_mask]

            else:
                p0 = self.p0[pred_mask]
                dt = self.dt[pred_mask]
                proba = p0 + dt * probab[pred_mask]
                dt = proba - p0
                self.proba[index] = proba
                self.dt = dt
                self.p0 = proba


            cp_[index]=probab[pred_mask]
            cr_[index]=y.r[:, -1][self.mask]
            cr=cr_
            cp=cp_
            r.append(cr)
            p.append(cp)

            self.x = y.get_items(mask=self.mask)
            self.top = self.top[pred_mask]
            self.indices = self.indices[pred_mask]
            i = i + 1

        self.p =np.array(p,dtype=np.float32)
        self.r=np.array(r,dtype=np.float32)
        return self.proba

    def get_new(self, x=np.array([]), tau=np.array([]), t=np.array([]), shape=np.array([])):
        #
        # [0:'ads', 1:'ads05',2:'ads1', 3'ads2', 4'ads3',
        # 5'ivl0', 6'ivl1', 7'ivl2',8'ivl3', 9'ivl4', 10'ivl5',
        # 11'nivl0', 12'nivl1', 13'nivl2', 14'nivl3', 15'nivl4',15'nivl5',
        # 17'wmean', 18'amean', 19'percent', 20'tau', 21'interval', 22'water', 23'length']
        # t-предыстория
        # tau -новые значения
        y = x.copy()
        y[:, self.col['tau']] = tau
        q = []
        j = 0
        # print(t.shape,shape.shape)
        for i in y:
            q.append(self.set_values(i, t[j], shape[j]))
            j = j + 1
        shape = shape + 1
        return y, np.array(q,dtype=object), shape

    def set_values(self, x=np.array([]), t=np.array([]), shape=np.array(1)):
        tau = x[self.col['tau']]
        q = np.append(t, tau)
        n = q.shape[0]
        for k in self.down_stairs.keys():
            mask = q >= tau - self.down_stairs[k]
            x[self.col[k]] = mask[mask == True].shape[0]
        x[self.col['ads']] = n
        wm = x[self.col['wmean']]
        am = x[self.col['amean']]
        w = x[self.col['water']]
        x[self.col['wmean']] = (wm * shape + w) / (shape + 1)
        x[self.col['amean']] = (am * shape + tau) / (shape + 1)
        if n > 5:
            x[self.col['ivl5']] = x[self.col['ivl5']] + 1
        else:
            x[self.col['ivl' + str(n - 2)]] = x[self.col['ivl' + str(n - 2)]] - 1
            x[self.col['ivl' + str(n - 1)]] = x[self.col['ivl' + str(n - 1)]] + 1
        arr=x[self.col['ivl0']:self.col['ivl5']+1]
        x[self.col['nivl0']:self.col['nivl5'] + 1]=np.cumsum(arr)

        return q




