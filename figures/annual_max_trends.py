annual_max = pd.read_feather('../output/ann_max_region'+'_window_'+str(24))

for region in range(0,16):
    plot = annual_max.T[region][2::]
    plt.scatter(range(1979,2024),plot)
    plt.show()


# how many storms above threshold occur each year for each region