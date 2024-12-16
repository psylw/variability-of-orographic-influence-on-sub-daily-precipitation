#%%
import pandas as pd

#%%
elev = pd.read_feather('../output/conus_elev')
aspect = pd.read_feather('../output/conus_aspect')
slope = pd.read_feather('../output/conus_slope')

add = pd.concat([slope,aspect],axis=1)

df_conus = pd.read_feather('../output/conus_ann_max').groupby(['latitude','longitude','year']).max().drop(columns='season').groupby(['latitude','longitude']).quantile(.5).reset_index()

df = pd.merge(df_conus,elev,on=['latitude','longitude'])
df = pd.merge(df,add,on=['latitude','longitude'])
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%%
for h in df.huc2.unique():
    
    test = df[(df.huc2==h)]
    X = test[[
        'elevation']]
    y = test['accum_24hr']


    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    # Evaluation

    print('huc2'+str(h))
    #print(mean_squared_error(y, y_pred))
    print(r2_score(y, y_pred))