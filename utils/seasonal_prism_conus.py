#%%
import xarray as xr
import glob
import matplotlib.pyplot as plt
import xesmf as xe
# open march 1981 through aug 2022
# change prism grid to conus404 grid
# change conus404 to daily accumulation

prism_files = glob.glob('../data/prism/seasonal/*.nc')

#conus_files = glob.glob('../data/conus404/*.nc')
conus_files = glob.glob('../data/conus404/PREC_ACC_NC_season/*.nc')
#conus_files = [file for file in conus_files if "1980" not in file and "1981_DJF" not in file]

#%%
df_conus = xr.open_dataset('../data/conus404/wrf2d_d01_2016_JJA.nc')
df_conus = df_conus.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))

df_prism = xr.open_dataset('../data/prism/seasonal\\PRISM_ppt_2021_DJF.nc')

regridder = xe.Regridder(df_prism, df_conus, "conservative")
#%%
c = []
for file in range(len(conus_files)):
    f1 = xr.open_dataset(conus_files[file])
    f1 = f1.sel(longitude = slice(-109,-104.005),latitude = slice(37,41))
    f1 = f1.sum(dim='time').PREC_ACC_NC
    f1['year_season'] = conus_files[file][-11:-3]

    c.append(f1)

# %%
p = []
for file in range(len(prism_files)):
    
    f2 = xr.open_dataset(prism_files[file])
    try:
        f2 = regridder(f2)

    except:
        # fix years where coordinates got messed up
        print(f2.time.dt.year.max())
        t1=f2.where(~f2.latitude.isin(df_prism.latitude),drop=True)
        t1=t1.where(~t1.longitude.isin(df_prism.longitude),drop=True)
        t1=t1.dropna(dim='time').assign_coords(latitude=df_prism.latitude, longitude=df_prism.longitude)

        t2=f2.where(f2.latitude.isin(df_prism.latitude),drop=True)
        t2=t2.where(t2.longitude.isin(df_prism.longitude),drop=True)
        t2=t2.dropna(dim='time')

        f2 = xr.concat([t1,t2],dim='time')
        f2 = regridder(f2)

    f2 = f2.sum(dim='time').band_data
    f2['year_season'] = prism_files[file][-11:-3]
    p.append(f2)


#%%
c = xr.concat(c,dim='year_season')
p = xr.concat(p,dim='year_season')
# %%
c.to_netcdf('../output/conus404_season_accum_temp.nc')
p.to_netcdf('../output/prism_season_accum_temp.nc')
# %%
