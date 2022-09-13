import pandas as pd
import numpy as np
import ps.app.calclib.pipeml as pml

path='D:\\ml\\'
file='gpn_raw_v5.csv'
dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
columns=['ID простого участка','D', 'L', 'S','Дата ввода', 'Состояние','Дата перевода в бездействие', 'Дата аварии', 'Наработка до отказа',
       'Адрес от начала участка', 'Обводненность','Дата окончания ремонта',
       'Адрес от начала участка_1', 'Длина ремонтируемого участка','Дата ремонта до аварии',
       'Адрес ремонта до аварии', 'Длина ремонта до аварии']
rdata=pd.read_csv(path+file,parse_dates=dates,infer_datetime_format=True, dayfirst=True)
ID=50007531
group=rdata[rdata['ID простого участка']==ID]
#mdate=group['Дата аварии'].max()
#group.loc[:,'Дата аварии']=mdate
#group.loc[:,repcolumns]=np.nan
pred=pml.predict(group,get='frame')
print()

#np.save(path+'pred.npy',pred.values)