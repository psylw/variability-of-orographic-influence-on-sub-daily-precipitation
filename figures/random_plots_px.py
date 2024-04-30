

# %%
# plot max intensity at each pixel
total_storms = grouped[grouped.year>2020]
all_years = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
plt.show()

for year in range(2021,2024):
    total_storms = grouped[grouped.year==year]
    total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
    plt.title(year)
    plt.show()
# %%
# max above threshold duration
total_storms = grouped[grouped.year>2020]
total_storms = total_storms.groupby(['latitude','longitude','storm_id']).count().reset_index()

total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
plt.title('all years')
plt.show()

for year in range(2021,2024):
    total_storms = grouped[grouped.year==year]
    total_storms = total_storms.groupby(['latitude','longitude','storm_id']).count().reset_index()

    total_storms = total_storms.groupby(['latitude','longitude']).max().unknown.to_xarray().plot()
    plt.title(year)
    plt.show()

# %%
# above threshold frequency
total_storms = grouped[grouped.year>2020].reset_index()
unique_values = total_storms.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
unique_values.to_xarray().plot()

times_perloc_perstorm = total_storms.groupby(['latitude','longitude','storm_id']).count().time.reset_index()
times_perloc_perstorm = times_perloc_perstorm.rename(columns={'time':'duration'})
total_storms = pd.merge(total_storms,times_perloc_perstorm,on=['latitude','longitude','storm_id'])
#%%
med_dur = times_perloc_perstorm.duration.quantile(.5)
med_int = total_storms.unknown.quantile(.9)

#%%
above_dur = total_storms.loc[(total_storms.duration>1)&(total_storms.unknown>40)]

unique_values = above_dur.groupby(['latitude','longitude'])['storm_id'].apply(lambda x: len(list(x.unique())))
unique_values.to_xarray().plot()
plt.title('all years')
plt.show()

# %%
