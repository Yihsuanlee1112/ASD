import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

df = pd.read_excel(
    "D:\\ASD\\【Vive】特徵計算\\Post\\ASD002_Post\\JointTrackingData.xlsx",
    usecols=["Value.pos.X", "Value.pos.Y", "Value.pos.Z"])
#print(df)
x = df['Value.pos.X']
y = df['Value.pos.Y']
z = df['Value.pos.Z']
#z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))

# Define the bin edges along each axis
xbins = np.linspace(x.min(), x.max(), 4)
ybins = np.linspace(y.min(), y.max(), 4)
zbins = np.linspace(z.min(), z.max(), 4)

# Compute the 3D histogram
hist, edges = np.histogramdd((x, y, z), bins=(xbins, ybins, zbins))

# Print the number of points in each bin
for i in range(len(xbins) - 1):
    for j in range(len(ybins) - 1):
        for k in range(len(zbins) - 1):
            print(f"Bin ({i}, {j}, {k}): {hist[i,j,k]}")

# Plot the 3D scatter plot with the bin edges
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, cmap='tab20', c=z, s=1, linewidths=None, edgecolors=None)
for i in range(len(xbins)):
    for j in range(len(ybins)):
        ax.plot([xbins[i], xbins[i]], [ybins[j], ybins[j]],
                [zbins[0], zbins[-1]],
                color='',
                linewidth=0.5)
        ax.plot([xbins[i], xbins[i]], [ybins[0], ybins[-1]],
                [zbins[j], zbins[j]],
                color='orange',
                linewidth=0.5)
        ax.plot([xbins[0], xbins[-1]], [ybins[j], ybins[j]],
                [zbins[i], zbins[i]],
                color='orange',
                linewidth=0.5)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Head Position')
# Show plot
plt.show()