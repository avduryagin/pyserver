import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from numpy.lib import recfunctions as rfn
from app.calclib import generator as gen, sets_methods as sm, metrics
from datetime import datetime

class repairs_map:
    def __init__(self):
        self.rmap=dict([])
        self.shape=0
        self.rep=np.array([])
    def fit(self,rep=np.array([])):
        self.rep=rep
        self.shape=rep.shape[0]
        if self.shape>0:
            self.get_repairs_map(self.shape-1)

    def get_repairs_map(self,k=0):
        if k == 0:
            X = self.rep[k, :].reshape(-1, 3)
            self.rmap.update({k: X})
            return X
        else:
            X = self.rep[k, :].reshape(3)
            L = self.get_repairs_map(k - 1)
            Y = sm.get_sets_residual(L, X)
            self.rmap.update({k:Y})
            return Y



def get_repairs_map(REP, k):
    if k == 0:
        X = REP[k, :].reshape(-1, 3)
        return X
    else:
        X = REP[k, :].reshape(3)
        L = get_repairs_map(REP, k - 1)
        Y = sm.get_sets_residual(L, X)
        return Y

def get_unical_repairs(data, dfield='repair_date',lfield='repair_length',afield='repair_address', scale=0,values=False):
    group = data[dfield].value_counts().keys()
    repairs = set()

    for rep in group:
        # print(rep)
        repgroup = data[data[dfield] == rep]
        for place in repgroup[afield].value_counts().keys():
            placegroup = repgroup[repgroup[afield] == place]
            for length in placegroup[lfield].value_counts().keys():
                if length >= scale:
                    repair = (rep, place, length)
                    repairs.add(repair)
    repairs = pd.DataFrame(list(repairs), columns=['Дата ремонта', 'Адрес', 'Длина']).sort_values(
        by='Дата ремонта').reset_index(drop=True)
    repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])
    if values:
        repairs['b'] = repairs['Адрес'] + repairs['Длина']
        values = repairs[['Адрес', 'b', 'Дата ремонта']].values
        return values
    return repairs

def get_raw_repairs(data, scale=0,values=False, rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии'):

    group=np.vstack((data[[rdfield,rxfield,rlfield]].values,data[[bdfield,bxfield,blfield]].values))
    mask=np.isnan(group[:,1].astype(np.float32))
    group=group[~mask]
    unique=np.unique(group[:,0])
    mask=np.ones(shape=group.shape[0],dtype=bool)
    repairs=[]

    for u in unique:
        index=np.where(group[mask,0]==u)[0]
        unique_place=np.unique(group[index,1])
        for up in unique_place:
            pindex=np.where(group[index,1]==up)[0]
            pindex_=index[pindex]
            unique_length=np.unique(group[pindex_,2])
            for ul in unique_length:
                lindex=np.where(group[pindex_,2]==ul)[0]
                lindex_=pindex_[lindex]
                #mask[lindex_]=False
                if ul>scale:
                    ub=up+ul
                    repair=(up,ub,u,ul)
                    repairs.append(repair)

    if values:
        if len(repairs)>0:
            repairs=[x[:3] for x in repairs]

        result = np.array(repairs).reshape(-1,3)

        return result
    else:
        repairs = pd.DataFrame(repairs, columns=['Адрес', 'b','Дата ремонта', 'Длина']).sort_values(
            by='Дата ремонта').reset_index(drop=True)
        repairs['Дата ремонта'] = pd.to_datetime(repairs['Дата ремонта'])

    return repairs[['Дата ремонта', 'Адрес', 'Длина']]

def get_merged_repairs(true, synthetic, epsilon=0.5,values=False):

    def split_repairs(true, synthetic, epsilon=0.5):
        result = np.array([]).reshape(-1, 3)
        if true.shape[0] == 0:
            return synthetic
        for synt in synthetic:
            r = true[0]
            delta = abs((synt[2] - r[2]) / np.timedelta64(1, 'Y'))
            if delta > epsilon:
                res = split_repairs(true[1:], synt.reshape(-1, 3), epsilon=epsilon)
            else:
                ispt = sm.interseption(r, synt)
                if ispt.shape[0] > 0:
                    residual = sm.get_sets_residual(synt.reshape(-1, 3), r)[:-1]
                    # print(residual)
                else:
                    residual = synt.reshape(-1, 3)
                    # print(residual)
                res = split_repairs(true[1:], residual, epsilon=epsilon)
            result = np.vstack((result, res))
        return result

    def merge_repairs(true, synthetic, epsilon=0.5):
        result = np.array([]).reshape(-1, 3)
        for s in synthetic:
            res = split_repairs(true, s.reshape(-1, 3), epsilon=epsilon)
            result = np.vstack((result, res))
        return result
    if values:
        merged = merge_repairs(true, synthetic, epsilon=epsilon)
        merged = np.vstack((merged, true))

        sargs=np.argsort(merged[:,2])
        return merged[sargs]

    else:
        columns = true.columns
        true['b'] = true['Адрес'] + true['Длина']
        synthetic['b'] = synthetic['Адрес'] + synthetic['Длина']
        true_reps = true[['Адрес', 'b', 'Дата ремонта']].values
        synt_reps = synthetic[['Адрес', 'b', 'Дата ремонта']].values
        merged = merge_repairs(true_reps, synt_reps, epsilon=epsilon)
        merged = np.vstack((merged, true_reps))
        df = pd.DataFrame(merged, columns=['Адрес', 'b', 'Дата ремонта'])
        df['Длина'] = df['b'] - df['Адрес']
        df = df[columns]
        df.sort_values(by='Дата ремонта', inplace=True)
        return df




def get_splited_by_repairs(data, index=np.array([],dtype=np.int32),ID='',delta=3,xfield ='Адрес от начала участка',dfield='Дата аварии',
              efield='Дата ввода',stfield='Состояние',outfield='Дата перевода в бездействие',rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии'):
    #group = data[data['ID простого участка'] == ID]
    group=data.loc[index]
    synthetic = get_unical_repairs(group,values=True)
    true=get_raw_repairs(group,values=True,rdfield=rdfield,rxfield=rxfield,rlfield=rlfield,bdfield=bdfield,bxfield=bxfield,blfield=blfield)
    rep=get_merged_repairs(true,synthetic,epsilon=0.5,values=True)
    #repairs['b'] = repairs['Адрес'] + repairs['Длина']
    #rep = repairs[['Адрес', 'b', 'Дата ремонта']].values
    rmap=repairs_map()
    rmap.fit(rep)
    # group['state']=True
    for i in np.arange(rep.shape[0]):
        X = rep[i]
        a = rep[i][0]
        b = rep[i][1]
        rd = rep[i][2]
        mask = (group[dfield] <= rd) & (
                    (group[xfield] <= b) & (group[xfield] >= a))
        mask1 = (group[dfield] > rd) & (
                    (group[xfield] <= b) & (group[xfield] >= a))
        indices = mask1[mask1 == True].index

        #data.loc[indexes, 'Дата ввода'] = rd
        data.loc[indices, 'a'] = a
        data.loc[indices, 'b'] = b
        data.loc[indices, 'new_id'] = str(ID) + '_' + str(i + 1)
        labels=mask[mask == True].index
        data.loc[labels, outfield] = rd
        data.loc[labels, stfield] = 'Бездействующий'
        repgroup = group[mask]
        if i > 0:
            A=rmap.rmap[i - 1]
            #A = get_repairs_map(rep, i - 1)
            T = sm.get_sets_residual(A, X.reshape(3), f=sm.interseption)[:-1]
            for t in T:
                submask = (repgroup[dfield] >= t[2]) & ((repgroup[xfield] <= t[1]) & (
                            repgroup[xfield] >= t[0]))
                mask = mask | submask
                indices1 = submask[submask == True].keys()
                dlt=(rd-t[2])/np.timedelta64(1,'Y')
                #print(dlt)
                if dlt>=delta:
                    data.loc[indices1, efield] = t[2]
                data.loc[indices1, outfield] = rd
                data.loc[indices1, stfield] = 'Бездействующий'

        #subgroup = repgroup[mask]
        #drop=mask[mask == False].index
        #group=group[~mask]

        group.drop(labels,axis=0,inplace=True)
    le = group[outfield] < group[dfield]
    group.loc[le,outfield]=np.nan
    empty = np.isnan(group[outfield])
    submask = ((group[stfield] == 'Бездействующий') | (group[stfield] == 'Демонтирован'))&empty
    indices1 = submask[submask == True].keys()
    rd = group[submask][dfield].max()
    data.loc[indices1, outfield] = rd
    if rep.shape[0]>0:
        A = rmap.rmap[rep.shape[0] - 1]
        #A = get_repairs_map(rep, rep.shape[0] - 1)
        for t in A:
            submask = (group[dfield] >t[2]) & ((group[xfield] <= t[1]) & (
                group[xfield] >= t[0])&((group[stfield]=='Бездействующий')|(group[stfield]=='Демонтирован')))
            indices1 = submask[submask == True].keys()
            rd=group[submask][dfield].max()
            dlt = (rd - t[2]) / np.timedelta64(1, 'Y')
            if dlt >= delta:
               data.loc[indices1, efield] = t[2]
            data.loc[indices1, outfield] = rd
            #data.loc[indexes1, 'Состояние'] = 'Бездействующий'
    #return data

def inscribing(data,*args,afield ='Наработка до отказа',xfield ='Адрес от начала участка',efield='Дата ввода',
               wfield ='Обводненность',lfield ='L',dfield='Дата аварии',by='ID простого участка',
               stfield='Состояние',outfield='Дата перевода в бездействие',rdfield='Дата окончания ремонта',
               rxfield='Адрес от начала участка_1',rlfield='Длина ремонтируемого участка',
               bdfield='Дата ремонта до аварии',bxfield='Адрес ремонта до аварии',blfield='Длина ремонта до аварии',
               cluster=None, repairs=None,split=None,today=None,**kwargs):
    #today = np.datetime64('2021-01-29')
    if today is None:
        today = datetime.now()
    #Поле в data:
    #afield ='Наработка до отказа'
    #xfield ='Адрес от начала участка'
    #wfield ='Обводненность'
    #lfield ='L'
    #dfields='Дата аварии'
    #stfield='Состояние'
    #outfield='Дата перевода в бездействие'
    #rdfield='Дата окончания ремонта'
    #rxfield='Адрес от начала участка_1'
    #rlfield='Длина ремонтируемого участка'

    #bdfield='Дата ремонта до аварии'
    #bxfield='Адрес ремонта до аварии'
    #blfield='Длина ремонта до аварии'

    #cluster={'epsilon':0.1,'r':0.5,'length':3000} - словарь с параметрами алгоритма кластеризации
    #repairs={'delta':1} - словарь с параметрами алгоритма агломерации ремонтов
    #split={'delta':3} -словарь с параметрами алгоритма формирование абстракных простых участков
    #by='ID простого участка' -признак группировки простых участков
    #today - Дата выгрузки данный из БД.
    #columns = ['ID простого участка', 'D', 'L', 'S', 'Дата ввода', 'Состояние', 'Дата перевода в бездействие',
               #'Дата аварии', 'Наработка до отказа',
               #'Адрес от начала участка', 'Обводненность', 'Дата окончания ремонта',
               #'Адрес от начала участка_1', 'Длина ремонтируемого участка', 'Дата ремонта до аварии',
               #'Адрес ремонта до аварии', 'Длина ремонта до аварии']


    def fill_wtp(data):
        data[wfield].fillna(0.,inplace=True)
        mask=data[wfield]>100
        data.loc[mask,wfield]=data.loc[mask,wfield]/10.
        mask = data[wfield]< 0
        data.loc[mask, wfield] = np.abs(data.loc[mask, wfield])

    if cluster is  None:
        cluster = {'epsilon': 0.1, 'r': 0.5, 'length': 3000}
    if repairs is None:
        repairs = {'delta': 1}
    if split is None:
        split = {'delta': 3}

    new_columns=['repair_date','repair_length','repair_address','repair_index','comment','cluster','a','b']
    n=data.columns.shape[0]
    for c in new_columns:
        data.insert(n,c,np.nan)
        n+=1

    #data.loc[:,'repair_date'] = np.nan
    #data.loc[:,'repair_length'] = np.nan
    #data.loc[:,'repair_address'] = np.nan
    #data.loc[:,'repair_index'] = np.nan
    #data.loc[:,'comment'] = np.nan
    #data.loc[:,'cluster']=np.nan
    #data.loc[:,'a'] = np.nan
    #data.loc[:,'b'] = np.nan


    for arg in args:
        data.loc[arg]=np.nan
    # отбрасываем записи с пустой координатой или наработкой до отказа
    mask=data[afield].isnull()|data[xfield].isnull()
    drop=mask[mask==True].index
    data.drop(drop,inplace=True)
    data.sort_values(by=afield,inplace=True)
    #data=data[~mask]
    # заполняем пропущенные значения обводненности, исправляем отрицательние или больше 100
    fill_wtp(data)
    # устанавливаем горизонт прогнозирования для действующих участков
    mask = data[stfield] == 'Действующий'
    data.loc[mask, outfield] = today
    data.loc[:,'new_id'] = data[by].astype('str')
    grouped = data.groupby(by).groups

    for group in grouped:
        index=grouped[group]
        length_approach(data,index,yfield=xfield,xfield=lfield)
        wtp_approach(data,index,xfield=afield,yfield=wfield)
        set_clusters(data,index,epsilon=cluster['epsilon'],r=cluster['r'],length=cluster['length'],xfield=xfield,afield=afield,lfield=lfield)
        set_repairs_by_clustering(data,index,delta=repairs['delta'],afield=afield,xfield=xfield,dfield=dfield)
        get_splited_by_repairs(data,index,ID=group,delta=split['delta'],xfield=xfield,dfield=dfield,efield=efield,stfield=stfield,outfield=outfield)
        #set_cover(data,c0=xfield,c1=afield,af='lbound',bf='rbound')

    data.loc[:,'L,м']=data['b']-data['a']
    data.loc[:,'Адрес от начала участка (new)']=data[xfield]-data['a']
    data.loc[:,'Наработка до отказа(new), лет']=(data[dfield]-data[efield])/np.timedelta64(1,'Y')
    data.loc[:,'getout'] = (data['Дата перевода в бездействие'] - data['Дата ввода']) / np.timedelta64(1, 'Y')
    data.loc[:,'to_out'] = (data['Дата перевода в бездействие'] - data['Дата аварии']) / np.timedelta64(1, 'Y')
    data.loc[:,'index']=data.index


def set_repairs_by_clustering(data=pd.DataFrame([]),index=np.array([],dtype=np.int32), delta=1,afield ='Наработка до отказа',xfield ='Адрес от начала участка',dfield='Дата аварии'):

    def get_cluster_shape(cluster,afield ='Наработка до отказа',xfield ='Адрес от начала участка',dfield='Дата аварии'):
        xmin = cluster[xfield].min()
        xmax = cluster[xfield].max()
        ymin = cluster[afield].min()
        ymax = cluster[afield].max()
        dmin = cluster[dfield].min()
        dmax = cluster[dfield].max()
        return (xmin, xmax), (ymin, ymax), (dmin, dmax)

    def get_cluster_interseption(cluster, data,afield ='Наработка до отказа',xfield ='Адрес от начала участка'):
        x, y, d = get_cluster_shape(cluster)
        mask = (data[xfield] >= x[0]) & (data[xfield] <= x[1]) & (
                    data[afield] > y[1])
        interseptions = data[mask]['cluster'].value_counts().keys()
        return interseptions

    pipe=data.loc[index]
    mask = pipe['cluster'] == -1
    alones = pipe[mask]
    tilde = 0
    for c in pipe[~mask]['cluster'].value_counts().keys():
        cluster = pipe[pipe['cluster'] == c]
        x, y, d = get_cluster_shape(cluster,afield=afield,xfield=xfield,dfield=dfield)
        rd = d[1]
        rl = x[1] - x[0]
        if rl < 2:
            rl = 2
        addr = x[0]
        interseption = get_cluster_interseption(cluster, pipe[~mask],afield=afield,xfield=xfield)
        comment = 0
        if len(interseption) > 0:
            for i in interseption:
                top = pipe[pipe['cluster'] == i]
                q, p, w = get_cluster_shape(top)
                if p[0] - y[1] <= delta:
                    comment = 1
                    break
        else:
            comment = 0
        alone = alones[(
                (alones[xfield] >= x[0]) & (alones[xfield] <= x[1]) & (
                alones[afield] < y[0]))]
        data.loc[cluster.index, ['repair_date', 'repair_address', 'repair_length', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]
        if alone.shape[0] > 0:
            data.loc[alone.index, ['repair_date', 'repair_address', 'repair_length', 'comment', 'repair_index']] = [rd,addr,rl,comment,tilde]

        tilde = tilde + 1
    #return data

def set_clusters(data=pd.DataFrame([]),index=np.array([],dtype=np.int32),epsilon=0.1,r=0.5,length=3000,by="ID простого участка",top_age=30,scaler=MinMaxScaler(),xfield='Адрес от начала участка',afield='Наработка до отказа',lfield='L'):
    #data['cluster']=np.nan
    scaler=scaler
    w=np.array([1,1],dtype=np.float32)
    group = data.loc[index]
    L = group.iloc[0][lfield]
    values = group[[xfield, afield]].values
    #values_=values
    #if L < length:
        #L = length
    add = np.array([length, top_age])
    values_ = np.vstack((values, add))
    scaler.fit(values_)
    tr = scaler.transform(values)
    Epsilon = r / scaler.data_range_[1]
    w[1] = (epsilon / Epsilon) ** 2
    metric = metrics.metrics(w)
    try:
        scan = DBSCAN(min_samples=2, eps=epsilon, metric=metric.euclidian).fit(tr)
        data.loc[group.index, 'cluster'] = scan.labels_
    except(ValueError):
        data.loc[group.index, 'cluster'] = 'error'


def wtp_approach(data=pd.DataFrame([]), index=np.array([],dtype=np.int32), xfield='Наработка до отказа', yfield='Обводненность'):
    #mask - бинарная маска "Обводненность"==0 & "ID простого участка" ==ID
    #data отсортирован по xfield
    mask=data.loc[index,yfield]==0
    if (mask[mask == True].shape[0] == 0) | (mask[mask == True].shape[0] == mask.shape[0]):
        return
    i = 0
    while i < mask.shape[0]:
        if mask.iloc[i]:
            if i > 0:
                a = mask.index[i - 1]
            else:
                a = mask.index[0]

            indices = []
            while mask.iloc[i]:
                indices.append(mask.index[i])
                if i < mask.shape[0] - 1:
                    i = i + 1
                else:
                    break

            if i == mask.shape[0] - 1:
                data.loc[indices, yfield] = data.loc[a, yfield]
            else:
                b = mask.index[i]
                q = data.loc[a, yfield]
                p = data.loc[b, yfield]
                x1 = data.loc[a, xfield]
                x2 = data.loc[b, xfield]
                x=np.array([x1,x2])
                y=np.array([q,p])
                delta=abs(x2-x1)


                if q > 0:
                    if delta>0:
                        line = sm.linear_transform(x=x, y=y)
                        data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line.value(x))
                    else:
                        line=sm.mean_approach(q,p)
                        data.loc[indices, yfield] = data.loc[indices, xfield].apply(lambda x: line(x))
                else:
                    data.loc[indices, yfield] = data.loc[b, yfield]
        i = i + 1


def length_approach(data=pd.DataFrame([]), index=np.array([],dtype=np.int32),xfield='L',yfield='Адрес от начала участка',afield='a',bfield='b'):
    L=data.loc[index,xfield].iloc[0]
    amax=data.loc[index,yfield].max()
    if amax>L:
        data.loc[index,xfield]=amax
    data.loc[index, afield] = 0
    data.loc[index, bfield] = data.loc[index, xfield]




class restrictions:
    def __init__(self):
        self.valid_indices=np.array([])
        self.map=np.array([])
        self.body=np.array([0,np.inf,0])
        self.valid_mask=np.array([],dtype=bool)
        self.cover=np.array([])

    def fit(self,x=np.array([]),rep=np.array([]),enter=np.datetime64('1978-01-01'),length=100,scale=2,mode='bw',size=100):

        def get_rep_cover(data, enter=np.datetime64('1978-01-01'), length=100, scale=2):
            bottom = np.array([0, length, 0])
            repairs_ = get_raw_repairs(data, scale=scale, values=True)
            if repairs_.shape[0] == 0:
                return bottom.reshape(-1, 3)
            repairs = get_repairs_map(repairs_, repairs_.shape[0] - 1)
            dtau = repairs[:, 2] - np.array(enter)
            tau = []
            for tau_ in dtau:
                tau.append(tau_ / np.timedelta64(1, 'Y'))
            tau = np.array(tau,dtype=np.float32).reshape(-1, 1)
            repairs_ = np.hstack((repairs[:, [0, 1]], tau))
            repairs_ = np.vstack((repairs_, bottom))
            djs = sm.get_disjoint_sets(repairs_, shape=3).astype(np.float32)
            return djs

        def set_valid_indices(data=np.array([[0, 0]]), restr=np.array([[0, np.inf, 0]])):
            def split(data=np.array([[0, 0]]), restr=np.array([[0, np.inf, 0]]), index=np.array([]),
                      indices=np.array([]), i=0):
                if i >= restr.shape[0]:
                    return
                a = restr[i, 0]
                b = restr[i, 1]
                d = restr[i, 2]
                mask = np.where((data[index, 0] >= a) & (data[index, 0] <= b) & (data[index, 1] >= d))[0]
                indices[index[mask]] = i
                index = np.delete(index, mask)
                i += 1
                split(data=data, restr=restr, index=index, indices=indices, i=i)

            indices = np.ones(shape=data.shape[0], dtype=np.int32) * -1
            index = np.arange(indices.shape[0])
            split(data=data, restr=restr, index=index, indices=indices, i=0)
            return indices

        def get(i=0):
            try:
                j=self.valid_indices[i]
                return self.map[j]
            except IndexError:
                return self.body

        def get_cover():
            self.cover=sm.cover(x[self.valid_mask],mode=mode,length=length,size=size,restrict=get)
            i=self.cover['i']
            ix=np.arange(x.shape[0])
            self.cover['i']=ix[self.valid_mask][i]

        self.body[1]=length
        self.map=get_rep_cover(data=rep,enter=enter,scale=scale,length=length)
        self.valid_indices=set_valid_indices(data=x,restr=self.map)
        self.valid_mask=~(self.valid_indices<0)
        self.valid_indices=self.valid_indices[self.valid_mask]
        get_cover()




class features:
    def __init__(self):
        self.data=np.array([])
        self.columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                            'Обводненность', 'getout','L,м', 'S','new_id','Дата аварии','index','to_out']
        self.ident='new_id'
        self.expand=False
        self.ints=np.array([100,150],dtype=np.int32)
        self.date=np.array([3],dtype=np.int32)
        self.steps=15
        self.epsilon=1/12.

        self.cl_features=['ads', 'ads05', 'ads1', 'ads2', 'ads3',
       'ivl0', 'ivl1', 'ivl2', 'ivl3', 'ivl4', 'ivl5', 'nivl0', 'nivl1',
       'nivl2', 'nivl3', 'nivl4', 'nivl5', 'wmean', 'amean', 'percent', 'tau',
        'water', 'length']
        self.clcols={w[1]:w[0] for w in zip(np.arange(len(self.cl_features)),self.cl_features)}

        self.reg_features=[]
        self.ClRe=gen.ClRe()
        self.s=np.array([])
        self.horizon = np.array([])

    def fit(self,xdata=pd.DataFrame([]),ident='new_id', expand=False, ints=np.array([100],dtype=np.int32),
            date=np.array([3],dtype=np.int32), steps=15, epsilon=1/12.,norm=True,mode='bw',restricts=True,drift=0.,**kwargs):
        self.ident=ident
        self.expand=expand
        self.ints=ints
        self.date=date
        self.steps=steps
        self.epsilon=epsilon
        self.raw=xdata
        self.reg_features = [str(x) for x in np.arange(self.steps)]
        #создание точек
        data=self.get_binary(self.raw,self.columns,date=self.date, ident=self.ident,expand=self.expand,ints=self.ints,steps=self.steps,epsilon=self.epsilon,mode=mode,restricts=restricts,drift=drift)

        if len(data)>0:
            self.data=np.vstack(data[0])
            self.cl=rfn.structured_to_unstructured(self.data[self.cl_features],dtype=np.float32).reshape(-1,len(self.cl_features))
            self.reg = rfn.structured_to_unstructured(self.data[self.reg_features], dtype=np.float32).reshape(-1,len(self.reg_features))
            #self.time_series = data[:, 1]
            self.time_series = np.array(data[1],dtype=object)
            self.s=self.data['s'].reshape(-1)
            self.top=self.data['top'].reshape(-1)
            self.shape = self.data['shape'].reshape(-1)
            self.horizon =self.data['horizon'].reshape(-1)
            self.features=self.data.dtype.names
            if norm:
                self.top=self.top/self.s
                self.horizon=self.horizon/self.s
                self.reg = np.divide(self.reg,self.s.reshape(-1,1))

            self.ClRe = gen.ClRe(c=self.cl, r=self.reg, s=self.s, t=self.time_series, shape=self.shape)






    def get_binary(self,xdata,columns,ident='ID простого участка',sortby='Наработка до отказа', expand=False, ints=np.array([100]), date=np.array([3]), steps=15, epsilon=1/12.,mode='bw',restricts=True,drift=0.):
        #mode - тип индексации
        def get_identity(data, date=1, a=0, b=1, index=-1, interval=100, steps=15, epsilon=1/12.):

            types = dict(
                names=['new_id', 'index', 'period', 'shape', 'Дата аварии', 'L,м', 'a', 'b', 'target', 'count', 'next',
                       'delta_next', 'delta',
                       'ads', 'ads05', 'ads1', 'ads2', 'ads3', 'ivl0', 'ivl1', 'ivl2', 'ivl3', 'ivl4', 'ivl5', 'nivl0',
                       'nivl1', 'nivl2', 'nivl3', 'nivl4', 'nivl5', 'wmean', 'amean', 'percent', 'tau', 'interval',
                       'water', 'x', 's', 'to_out','length','top','horizon'],
                formats=['U25', np.int32, np.int8, np.int32, 'datetime64[s]', np.float, np.float, np.float, np.float,
                         np.float,
                         np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float,
                         np.float, np.float, np.float,
                         np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float,
                         np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float,np.float, np.float,np.float])

            def sparse(rs=np.array([]), epsilon=0.1):
                l = []
                prev = rs[0]
                x = None
                a = prev + epsilon
                l.append(prev)
                i = 1
                while i < rs.shape[0]:
                    x = rs[i]
                    if x > a:
                        t = rs[i]
                        a = x + epsilon
                        if t != prev:
                            prev = t
                            l.append(t)
                    i = i + 1
                if (x is not None) & ~(x == np.array(l[-1])):
                    l.append(x)
                return np.array(l)


            def get_horizontal_counts(data=np.array([]), interval=100, L=100):
                mask = np.ones(data.shape[0], dtype=bool)
                intervals = []
                i = 0
                while mask.shape[0] > 0:
                    y = data[-1]
                    a = y - interval
                    b = y + interval
                    if a < 0:
                        a = 0
                    if b > L:
                        b = L
                    res = np.array([a, b])
                    if i == 0:
                        intervals.append((0, 0, 0))
                        i = i + 1
                    for ivl in intervals:
                        if res.shape[0] > 0:
                            res = sm.residual(res, ivl, shape=2).reshape(-1)
                    if res.shape[0] > 0:
                        submask = (data >= res[0]) & (data <= res[1])
                        res = np.append(res, submask[submask == True].shape[0])
                        intervals.append(res)
                        data = data[~submask]
                        mask = mask[~submask]
                    else:
                        mask[-1] = False
                        data = data[mask]
                        mask = mask[mask]
                return np.array(intervals[1:])

            columns = np.arange(steps)
            columns = [str(x) for x in columns]
            for c in columns:
                types['names'].append(c)
                types['formats'].append(np.float)
            dtype = np.dtype(types)
            identity = np.empty(shape=(1), dtype=dtype)
            step = dict({'ads05': 0.5, 'ads1': 1., 'ads2': 2., 'ads3': 3.})
            tau = data[index, 0]
            x = data[index, 1]
            out = data[index, 3]
            length = data[index, 4]
            s = data[index, 5]
            id = data[index, 6]
            adate = data[index, 7]
            i = data[index, 8]
            to_out = data[index, 9]
            identity['new_id'] = id
            identity['s'] = s
            identity['to_out'] = to_out
            identity['tau'] = tau
            identity['interval'] = interval
            identity['index'] = i
            identity['period'] = date
            identity['Дата аварии'] = adate
            identity['water'] = data[index, 2]
            identity['L,м'] = length
            identity['a'] = a
            identity['b'] = b
            identity['length'] = b-a
            identity['x'] = x
            identity['top'] = min(tau+date,tau+to_out)
            identity['horizon']=tau+date

            mtau=masked(index)
            mask = data[:, 0] <= tau
            hormask = mask
            if mtau>tau:
                hormask = data[:, 0] <= mtau



            identity['shape'] = hormask[hormask == True].shape[0]
            mask1 = (data[:, 1] >= a) & (data[:, 1] <= b)
            xmask = mask1 & mask
            ads = xmask[xmask == True].shape[0]
            dt = np.nan
            prev = 0
            if ads > 1:
                prev = data[xmask, 0][-2]

            dt = tau - prev
            identity['delta'] = dt
            identity['ads'] = ads

            sparsed = sparse(data[:, 0][xmask], epsilon=epsilon)[-steps:]
            for t in np.arange(1, steps + 1):
                if -t >= -sparsed.shape[0]:
                    identity[columns[-t]] = sparsed[-t]
                else:
                    identity[columns[-t]] = 0

            for k in step.keys():
                # dlt=tau-step[k]
                substep = data[:, 0] >= tau - step[k]
                smask = substep & xmask
                identity[k] = smask[smask == True].shape[0]
            ivls = get_horizontal_counts(data[:, 1][hormask], interval=interval, L=length)
            res = ivls[:, 1] - ivls[:, 0]
            identity['percent'] = res.sum() / length
            w_mean = data[:, 2][mask].mean()
            a_mean = data[:, 0][mask].mean()
            identity['wmean'] = w_mean
            identity['amean'] = a_mean
            ivl_counts = ivls[:, 2].astype(int)
            for ii in np.arange(6):
                if ii == 5:
                    mask3 = ivl_counts >= ii + 1
                    mask4 = ivl_counts >= 0
                else:
                    mask3 = ivl_counts == ii + 1
                    mask4 = ivl_counts <= ii + 1
                identity['ivl' + str(ii)] = mask3[mask3 == True].shape[0]
                identity['nivl' + str(ii)] = mask4[mask4 == True].shape[0]
            tmask = mask1 & (~mask)
            top = tau + date
            mask2 = data[:, 0] <= top
            ymask = tmask & mask2
            target = np.nan
            next = np.nan
            delta = np.nan

            identity['next'] = next
            identity['delta_next'] = delta
            dic = {0: 8. / 12., 1: 7. / 12., 2: 5. / 12., 3: 4. / 12., 4: 3. / 12., 5: 3. / 12, 6: 2. / 12.,
                   7: 2. / 12., }
            count = ymask[ymask == True].shape[0]
            if count > 0:
                arange = np.arange(tmask.shape[0])
                inext = arange[tmask][0]
                next = data[inext, 0]
                delta = next - tau
                identity['next'] = next
                identity['delta_next'] = delta

            if top <= out:
                if count > 0:
                    target = 1
                else:
                    target = 0
            else:
                if count > 0:
                    target = 1
                else:
                    target = np.nan
                    count = np.nan

            identity['target'] = target
            identity['count'] = count
            ts=data[:, 0][xmask].astype(float)
            return identity,ts
            # return identity, sparsed




        xdata.sort_values(by=sortby, inplace=True)
        aggdata = xdata.groupby(ident)
        npints = np.array(ints) * 2
        L = []

        for i, group in enumerate(aggdata):
            Length = group[1]['L'].iloc[0]
            data = group[1][['Адрес от начала участка','Наработка до отказа']].values
            mask = np.where(npints <= Length)[0]
            k = mask.shape[0]
            if k>0:
                for teta in ints:
                    if restricts:
                        restr=restrictions()
                        restr.fit(x=data,rep=group[1],enter=group[1]['Дата ввода'].min(),size=teta,length=Length,mode=mode)
                        val=restr.cover
                    else:
                        val=sm.cover(data,mode=mode,length=Length,size=teta,c0=0,c1=1)
                    index=group[1].index[val['i']]
                    subgroups=group[1].groupby('new_id')
                    uniques=np.unique(group[1].loc[index]['new_id'].values)

                    for ID in uniques:
                        subgroup=subgroups.groups[ID]
                        #length,x=group[1].iloc[i][['L,м','Адрес от начала участка']]
                        length=group[1].loc[subgroup[0],'L,м']
                        mask= np.where(npints <= length)[0]
                        n = mask.shape[0]
                        if n>0:
                            subindex=np.where(np.isin(subgroup,index))[0]
                            #sub - x,tau координаты ID
                            sub=group[1].loc[subgroup,columns].values
                            #максимальное значение возраста аварии
                            taumax=sub[:,0].max()
                            low=taumax*(1-drift)
                            drift_mask=sub[:,0]>=low
                            masked=sm.masked(sub[:,0],drift_mask,taumax)

                            for j in subindex:
                                s=subgroup[j]
                                v=index.get_loc(s)
                                x=group[1].loc[s,'a']
                                if teta * 2 <= length:
                                    a_=val['a'][v]-x
                                    b_=val['b'][v]-x
                                    #a_,b_=val[v][[1,2]]-x
                                    bound=sm.interseption((a_,b_),(0,length),shape=2).reshape(-1)
                                    if bound.shape[0]>0:
                                        if bound[1]-bound[0]>=teta:
                                            for d in date:
                                                tensor,ts = get_identity(sub, date=d, a=bound[0], b=bound[1], index=j,
                                                                            interval=teta, epsilon=epsilon, steps=steps)
                                                if tensor is not None:
                                                    L.append((tensor,ts))
                                                else:
                                                    print('empty id', group[0])

        transpose=[list(x) for x in zip(*L)]
        return transpose






