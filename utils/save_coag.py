#%%
import glob
import pandas as pd

# open all gage data
files = glob.glob('../data/coagment/*.csv')

coag = []
for file in files[1::]:
    coag.append(pd.read_csv(file, skiprows=1))
coag = pd.concat(coag).rename(columns={'mm':'accum'})
#%%
# assign region
df = pd.read_csv("../data/coagment/metadata.csv")

df = df.rename(columns={'Station':'id','Latitude (degN)':'latitude','Longitude (degE)':'longitude'})

df = df.drop(columns=['Name', 'Location','Elevation (m)', 'Anemometer Height (m)', 'Active', 'Irrigation', 'Timestep (s)', 'Network','First Observation', 'Last Observation'])
#%%

coag = pd.merge(coag,df,on=['id'])
coag = coag.rename(columns={'date time':'time'})
#%%
coag['time'] = pd.to_datetime(coag['time'], format="%m/%d/%Y %H:%M", errors='coerce')
coag = coag[coag.time.dt.year>=2016]
coag = coag[coag.longitude<-104.005]
#%%
import numpy as np
import matplotlib.pyplot as plt
# basic QC
coag = coag[coag.accum>=0].reset_index()

test = coag.groupby(['latitude','longitude']).agg(list).accum.reset_index()
print(len(test))
t=[len(np.unique(test.iloc[i].accum)) for i in test.index]
test['unique'] = t
bad = test[test['unique']<10]

coag = coag[~((coag.latitude.isin(bad.latitude))&(coag.longitude.isin(bad.longitude)))]

#%%
test2 = coag.groupby([coag.time.dt.year,'latitude','longitude']).max()
test2 = test2[test2.accum==0]

coag = coag[~((coag.id.isin(test2.id)))]
#%%

coag['year'] = coag.time.dt.year
coag_24=[]
for year in range(2016,2023):
    coag_24.append(coag[coag.year==year].groupby(['latitude','longitude','time']).max().accum.to_xarray().rolling(time=24).sum().to_dataframe())
coag_24=pd.concat(coag_24).dropna()
#%%
coag_24 = coag_24[coag_24.accum>0.01].reset_index()

coag = coag[coag.accum>0.01]
#%%
coag.to_feather('../output/coag_1')
coag_24.to_feather('../output/coag_24')
