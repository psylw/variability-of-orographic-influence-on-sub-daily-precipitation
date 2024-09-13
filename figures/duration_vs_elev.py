# for each region, for all intensity windows, plot elevation vs. max annual annual duration
#%%
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
# compare median max precip at different thresholds at different windows
for q in np.arange(0.125,0.625,0.125):
    print(q)
    for window in (1,3,12,24):
        print(window)
        files = glob.glob('../../output/duration_'+str(window)+'/*')

        df_all_years = []
        for file in files:

            df = pd.read_feather(file)
            df = df[(df.quant==q)]
            df_all_years.append(df)

        df_all_years = pd.concat(df_all_years)

        df_all_years.groupby(['latitude','longitude']).median().max_precip.to_xarray().plot()

        plt.show()
#%%
# compare median duration at different thresholds at different windows
for q in np.arange(0.125,0.625,0.125):
    print(q)
    for window in (1,3,12,24):
        print(window)
        files = glob.glob('../output/duration_'+str(window)+'/*')

        df_all_years = []
        for file in files:

            df = pd.read_feather(file)
            df = df[(df.quant==q)]
            df_all_years.append(df.groupby(['latitude','longitude']).max().APCP_surface.reset_index())

        df_all_years = pd.concat(df_all_years)

        df_all_years.groupby(['latitude','longitude']).median().APCP_surface.to_xarray().plot()

        plt.show()
#%%
elev = pd.read_feather('../output/elevation')
window = 24
for q in np.arange(0.125,0.625,0.125):
    print(q)
    #for window in (1,3,12,24):
        #print(window)
    files = glob.glob('../output/duration_'+str(window)+'/*')

    df_all_years = []
    for file in files:

        df = pd.read_feather(file)
        df = df[(df.quant==q)]
        df = pd.merge(df,elev[['latitude','longitude','elevation_category']],on=['latitude','longitude'])

        df_all_years.append(df.groupby(['elevation_category','region']).count().latitude.reset_index())

    df_all_years = pd.concat(df_all_years)

    plt.figure(figsize=(10, 8))
    sns.boxplot(data = df_all_years, x = 'region', y = 'latitude', hue = 'elevation_category')
    plt.show()