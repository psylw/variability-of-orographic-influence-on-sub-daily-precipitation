#%%
###############################################################################
# save coordinates for each storm to calculate storm stats 1 storm at a time.
###############################################################################
import xarray as xr
import os
import glob

storm_folder = os.path.join('..', '..','..','data',"storm_stats")

file_storm = glob.glob(storm_folder+'//'+'*.nc')
#%%
# save coordinates, DELETE other storm files #MOVE TO SAVE_STORMID.PY
# 12 min/file, takes most memory
for file in file_storm:
    storm = xr.open_dataset(file,chunks={'time': '10MB'})
    storm = storm.where(storm>0,drop=True)
    midpoint = len(storm.time)//2
    for i in range(2):
        name = '//'+file[-28:-3]+'_coord'+str(i)
        if i == 0:
            s = storm.isel(time = slice(None,midpoint))
            s = s.to_dataframe()
            s = s.loc[s.storm_id>0]
            s = s.reset_index().groupby(['storm_id']).agg(list)
            s = s.reset_index()

            s.to_feather(storm_folder+name)
        else:
            s = storm.isel(time = slice(midpoint,None))
            s = s.to_dataframe()
            s = s.loc[s.storm_id>0]
            s = s.reset_index().groupby(['storm_id']).agg(list)
            s = s.reset_index()

            s.to_feather(storm_folder+name)
# %%
