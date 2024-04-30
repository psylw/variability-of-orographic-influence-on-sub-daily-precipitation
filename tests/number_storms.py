
import glob

df = pd.read_feather('../output/count_duration_threshold_px')
df=df.loc[(df.latitude!=40.624999999999282)&(df.longitude!=253.33499899999782)]
num_stums = []
for year in range(2015,2024):
    for month range(5,10):
        sample = df.loc[(df.month==month)&(df.year==year)]
        num_stums.append(len(sample.storm_id.unique()))
np.sum(num_stums)

files = glob.glob('..\\output\\*tes')

all = []
for file in files:
       df = pd.read_feather(file)
       all.append(df)

df = pd.concat(all).reset_index().fillna(0) 

len(df)