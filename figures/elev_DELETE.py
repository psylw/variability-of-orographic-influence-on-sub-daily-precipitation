#%%
import xarray as xr
import xesmf as xe
import rioxarray
import matplotlib.pyplot as plt

df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

source_raster = rioxarray.open_rasterio("../../../data/elev_data/CO_SRTM1arcsec__merge.tif")


#%%
test2 = df_conus.sel(longitude = slice(-109.1,-108.5),latitude = slice(37,37.5))
test = source_raster.sel(x = slice(-109.1,-108.5),y = slice(37.5,37))
test = test.rename({'x': 'lon','y':'lat'})

#%%
regridder_aorc = xe.Regridder(test, test2, "conservative")

precip = regridder_aorc(test)
test.sel(band=1).plot()
plt.show()
precip.sel(band=1).plot()
plt.show()
# %%
regridder_aorc = xe.Regridder(test, test2, "bilinear")

precip2 = regridder_aorc(test)
test.sel(band=1).plot()
plt.show()
precip2.sel(band=1).plot()
plt.show()

(precip.sel(band=1)-precip2.sel(band=1)).plot(vmin=-50,vmax=50)
plt.show()
# %%
import pandas as pd
df_elev = pd.read_feather('../output/conus_elev')

# %%
