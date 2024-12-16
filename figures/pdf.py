# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
import numpy as np
from scipy.stats import gaussian_kde
#%%
coag1 = pd.read_feather('../output/coag_1')
coag24 = pd.read_feather('../output/coag_24')

coag_coord = coag1.groupby(['latitude','longitude']).max().reset_index()

elev = pd.read_feather('../output/'+'conus'+'_elev')
elev = elev.groupby(['latitude', 'longitude']).max().huc2.to_xarray()

huc = []
for coord in coag_coord.index:
    data = {'latitude':coag_coord.latitude[coord],
    'longitude':coag_coord.longitude[coord],
    'huc2':elev.sel(latitude = coag_coord.latitude[coord],longitude= coag_coord.longitude[coord],method='nearest').values}
    huc.append(data)

huc = pd.DataFrame(huc)

coag1 = pd.merge(coag1,huc,on=['latitude','longitude'])
coag24 = pd.merge(coag24,huc,on=['latitude','longitude'])
#%%
aorc1 = pd.read_feather('../output/aorc_atgage')
aorc1 = aorc1[aorc1.accum_1hr>=.13][['time', 'latitude', 'longitude', 'accum_1hr']]
conus1 = pd.read_feather('../output/conus_atgage')
conus1 = conus1[conus1.accum_1hr>=.13][['time', 'latitude', 'longitude', 'accum_1hr']]
mrms1 = pd.read_feather('../output/mrms_atgage')
mrms1 = mrms1[mrms1.accum_1hr>=.13][['time', 'latitude', 'longitude', 'accum_1hr']]

aorc24 = pd.read_feather('../output/aorc_atgage')
aorc24 = aorc24[aorc24.accum_24hr>=.13][['time', 'latitude', 'longitude', 'accum_24hr']]
conus24 = pd.read_feather('../output/conus_atgage')
conus24 = conus24[conus24.accum_24hr>=.13][['time', 'latitude', 'longitude', 'accum_24hr']]
mrms24 = pd.read_feather('../output/mrms_atgage')
mrms24 = mrms24[mrms24.accum_24hr>=.13][['time', 'latitude', 'longitude', 'accum_24hr']]

aorc1 = pd.merge(aorc1,huc,on=['latitude','longitude'])
conus1  = pd.merge(conus1,huc,on=['latitude','longitude'])
mrms1 = pd.merge(mrms1,huc,on=['latitude','longitude'])
aorc24 = pd.merge(aorc24,huc,on=['latitude','longitude'])
conus24  = pd.merge(conus24,huc,on=['latitude','longitude'])
mrms24 = pd.merge(mrms24,huc,on=['latitude','longitude'])
# %%
huc = [14,10,13,11]

fig, axes = plt.subplots(2, 2, figsize=(10, 8),sharex=True,sharey=True)

# Flatten the axes array for easier iteration
axes_flat = axes.flatten()

# Loop to fill each subplot with the corresponding function
for i, ax in enumerate(axes_flat):
    
    h = huc[i]

    kde1 = gaussian_kde(coag1[coag1.huc2==h]['accum'])
    kde2 = gaussian_kde(aorc1[aorc1.huc2==h]['accum_1hr'])
    kde3 = gaussian_kde(mrms1[mrms1.huc2==h]['accum_1hr'])
    kde4 = gaussian_kde(conus1[conus1.huc2==h]['accum_1hr'])

    x_values = np.linspace(0, 20, 100)

    pdf_values1 = kde1(x_values)
    pdf_values2 = kde2(x_values)
    pdf_values3 = kde3(x_values)
    pdf_values4 = kde4(x_values)

    kde1a = gaussian_kde(coag24[coag24.huc2==h]['accum'])
    kde2a = gaussian_kde(aorc24[aorc24.huc2==h]['accum_24hr'])
    kde3a = gaussian_kde(mrms24[mrms24.huc2==h]['accum_24hr'])
    kde4a = gaussian_kde(conus24[conus24.huc2==h]['accum_24hr'])

    x_values1 = np.linspace(0, 25, 100)

    pdf_values1a = kde1a(x_values1)
    pdf_values2a = kde2a(x_values1)
    pdf_values3a = kde3a(x_values1)
    pdf_values4a = kde4a(x_values1)

    ax.plot(x_values, pdf_values1, label='CoAgMET 1-hr', color='black')
    ax.plot(x_values, pdf_values2, label='AORC 1-hr', color=sns.color_palette("colorblind")[1])
    ax.plot(x_values, pdf_values3, label='MRMS 1-hr', color=sns.color_palette("colorblind")[0])
    ax.plot(x_values, pdf_values4, label='CONUS404 1-hr', color=sns.color_palette("colorblind")[2])

    ax.plot(x_values1, pdf_values1a, label='CoAgMET 24-hr', color='black',linestyle="--")
    ax.plot(x_values1, pdf_values2a, label='AORC 24-hr', color=sns.color_palette("colorblind")[1],linestyle="--")
    ax.plot(x_values1, pdf_values3a, label='MRMS 24-hr', color=sns.color_palette("colorblind")[0],linestyle="--")
    ax.plot(x_values1, pdf_values4a, label='CONUS404 24-hr', color=sns.color_palette("colorblind")[2],linestyle="--")

    #plt.xlabel('accum, mm')
    #plt.ylabel('density')

    ax.set_yscale('log')
    ax.set_title(f"HUC2: {h}", loc='center', pad=-100, fontsize=14)
    ax.set_ylim(1e-3, 1e0)
    #ax.legend()
    ax.set_xlim(0,25)
    if i == 1:
        ax.legend()

fig.supxlabel('accum, mm', fontsize=14)
fig.supylabel('density', fontsize=14)

plt.tight_layout()
plt.show()

fig.savefig("../figures_output/pdf.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')