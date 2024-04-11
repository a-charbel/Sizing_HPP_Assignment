#%%
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
#%%

# read in the data from the first file
data_solar = pd.read_csv('aveS_1.csv')
data_wind = pd.read_csv('aveW_1.csv')

# multiply all non-time values by a factor to generate another dataset
data_solar['Psolar'] *= 1
data_wind['Pwind'] *= 0.85

# write the modified data to a new file
data_solar.to_csv('aveS_4.csv', index=False)
data_wind.to_csv("aveW_4.csv", index=False)


# %%
