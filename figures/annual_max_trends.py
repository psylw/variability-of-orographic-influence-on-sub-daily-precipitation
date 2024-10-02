#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
window = 1
annual_max = pd.read_feather('../output/aorc_ann_max_region'+'_window_'+str(window)).iloc[:,2:]
annual_max_nd = pd.read_feather('../output/nldas_ann_max_region'+'_window_'+str(window)).iloc[:,2:]
annual_max_c = pd.read_feather('../output/conus_ann_max_region'+'_window_'+str(window)).iloc[:,2:]
#%%


annual_max = annual_max.stack().reset_index().rename(columns={'level_0':'region','level_1':'year',0:'max'})
annual_max['dataset'] = 'aorc'

annual_max_nd = annual_max_nd.stack().reset_index().rename(columns={'level_0':'region','level_1':'year',0:'max'})
annual_max_nd['dataset'] = 'nldas'

annual_max_c = annual_max_c.stack().reset_index().rename(columns={'level_0':'region','level_1':'year',0:'max'})
annual_max_c['dataset'] = 'conus'
#%%
df = pd.concat([annual_max,annual_max_nd,annual_max_c])


# %%
plt.figure(figsize=(10, 6)) 
sns.lineplot(data=df,x='year',y='max',hue='dataset')
plt.hlines(annual_max['max'].quantile(.3),xmin='year2002',xmax='year2022',colors='blue')


plt.hlines(annual_max_nd['max'].quantile(.3),xmin='year2002',xmax='year2022',colors='orange')
plt.hlines(annual_max_c['max'].quantile(.3),xmin='year2002',xmax='year2022',colors='green')
# %%
sns.kdeplot(
    df,
    x="max", hue="dataset",
    


)