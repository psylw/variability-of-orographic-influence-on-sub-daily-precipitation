import matplotlib.pyplot as plt

m=m.sel(time=temp.time,latitude=temp.latitude,longitude=temp.longitude).unknown

for i in range(len(m.time)):
    m.isel(time=i).plot()
    plt.show()