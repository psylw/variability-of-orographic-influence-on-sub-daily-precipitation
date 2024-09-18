#%%
import pandas as pd
import matplotlib.pyplot as plt

annual_max = pd.read_feather('../output/ann_max_region'+'_window_'+str(1))
annual_max_nd = pd.read_feather('../output/nldas_ann_max_region'+'_window_'+str(1))
annual_max_c = pd.read_feather('../output/conus_ann_max_region'+'_window_'+str(1))
#%%
for region in range(0,16):
    print(region)
    plot = annual_max.T[region][2::]
    plot2 = annual_max_nd.T[region][2::]
    plot3 = annual_max_c.T[region][2::]

    plt.plot(range(2002,2021),plot[2002-1979:-3],label='aorc')
    plt.plot(range(2002,2021),plot2,label='nldas')
    plt.plot(range(2002,2021),plot3[:-2],label='conus404')
    plt.legend()
    plt.show()


# how many storms above threshold occur each year for each region
# %%
