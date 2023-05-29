import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from datetime import datetime

df = pd.read_excel(
    'D:\\ASD\\【Vive】特徵計算\\Post\\ASD002_Post\\EEGFeatureIndexesData.xlsx',
    usecols=[
        "Value.EEGSensorType", "Value.DELTA_Data", "Value.THETA_Data",
        "Value.ALPHA_Data", "Value.BETA_Data", "Value.GAMMA_Data",
        "Value.Timestamp"
    ])
# print(df)
d = df['Value.DELTA_Data']
t = df['Value.THETA_Data']
a = df['Value.ALPHA_Data']
b = df['Value.BETA_Data']
g = df['Value.GAMMA_Data']
s = df['Value.EEGSensorType']
timestamp = df['Value.Timestamp']
# dt_obj = datetime.fromisoformat(timestamp)
# str_date_time = datetime.strftime("%d-%m-%Y, %H:%M:%S")
# Define the subplots layout
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))

# Plot data on each subplot
axes[0].plot(d, linewidth=0.1)
axes[0].set_title('Delta')
axes[1].plot(t, linewidth=0.1)
axes[1].set_title('Theta')
axes[2].plot(a, linewidth=0.1)
axes[2].set_title('Alpha')
axes[3].plot(b, linewidth=0.1)
axes[3].set_title('Beta')
axes[4].plot(g, linewidth=0.1)
axes[4].set_title('Gamma')
# Set overall figure title and adjust layout
fig.suptitle('EEGFeatureData')
# Set common x and y labels for all subplots
fig.text(0.5, 0.04, 'time(s)', ha='center', va='center')
fig.text(0.06, 0.5, ' ', ha='center', va='center', rotation='vertical')
# Adjust subplot spacing
plt.subplots_adjust(hspace=0.5)
# Show plot
plt.show()