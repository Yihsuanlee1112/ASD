import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

df = pd.read_excel(
    'D:\\ASD\\【Vive】特徵計算\\Post\\ASD002_Post\\JointTrackingData.xlsx',
    usecols=["Value.qua.X", "Value.qua.Y", "Value.qua.Z", "Value.qua.W"])
#print(df)
x = df['Value.qua.X']
y = df['Value.qua.Y']
z = df['Value.qua.Z']
w = df['Value.qua.W']
# Create quaternion array
q = np.zeros((len(x), 4))
q[:, 0] = x
q[:, 1] = y
q[:, 2] = z
q[:, 3] = w

# Compute rotation matrix from quaternion
R = np.zeros((len(q), 3, 3))
for i in range(len(q)):
    q_i = q[i]
    q_i /= np.linalg.norm(q_i)  # Ensure unit quaternion
    q_w, q_x, q_y, q_z = q_i
    R[i] = np.array([[
        1 - 2 * q_y**2 - 2 * q_z**2, 2 * q_x * q_y - 2 * q_z * q_w,
        2 * q_x * q_z + 2 * q_y * q_w
    ],
                     [
                         2 * q_x * q_y + 2 * q_z * q_w,
                         1 - 2 * q_x**2 - 2 * q_z**2,
                         2 * q_y * q_z - 2 * q_x * q_w
                     ],
                     [
                         2 * q_x * q_z - 2 * q_y * q_w,
                         2 * q_y * q_z + 2 * q_x * q_w,
                         1 - 2 * q_x**2 - 2 * q_y**2
                     ]])

# Create quaternion plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(R[:, 0, 0], R[:, 1, 0], R[:, 2, 0], marker='*', s=1)  #x
ax.scatter(R[:, 0, 1], R[:, 1, 1], R[:, 2, 1], marker='+', s=1)  #y
ax.scatter(R[:, 0, 2], R[:, 1, 2], R[:, 2, 2], marker='x', s=1)  #z

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Head rotation')
# Show plot
plt.show()