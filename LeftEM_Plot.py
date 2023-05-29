import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
# 左眼資料其實是右眼
def read_data(filename):
    df = pd.read_excel(
        filename,
        usecols=[
            "Value.gaze_origin_mm.X", "Value.gaze_origin_mm.Y", "Value.gaze_origin_mm.Z",
            "Value.pupil_position_in_sensor_area.x", "Value.pupil_position_in_sensor_area.y",
            "Value.eye_openness"
        ])

    return df

def calculate_eye_openness_stats(df):
    eye_openness_mean = np.mean(df['Value.eye_openness'])
    eye_openness_std = np.std(df['Value.eye_openness'])
    return eye_openness_mean, eye_openness_std

def calculate_pupil_path_length(df):
    positions = df[['Value.pupil_position_in_sensor_area.x',
                    'Value.pupil_position_in_sensor_area.y']].to_numpy()
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    pupil_path_length = np.sum(distances)
    return pupil_path_length

def create_feature_dataframe(eye_openness_mean, eye_openness_std, pupil_path_length):
    result = pd.DataFrame({
        'eye_openness_mean': [eye_openness_mean],
        'eye_openness_std': [eye_openness_std],
        'pupil_path_length': [pupil_path_length]
    })
    return result

def plot_3d_scatter_with_bins(x, y, z):
    # Define the bin edges along each axis
    xbins = np.linspace(x.min(), x.max(), 3)
    ybins = np.linspace(y.min(), y.max(), 3)
    zbins = np.linspace(z.min(), z.max(), 3)
    #Compute the 3D histogram
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
                    color='orange',
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
    ax.set_title('Right Eye gaze point')  # 左眼資料其實是右眼
    # Show plot
    plt.show()

if __name__ == '__main__':
    df = read_data("D:\\ASD\\【Vive】特徵計算\Pre\\ASD001_Pre\\puzzle\\LabLeftEyeData_Puzzle.xlsx")
    x = df['Value.gaze_origin_mm.X']
    y = df['Value.gaze_origin_mm.Y']
    z = df['Value.gaze_origin_mm.Z']
    # # z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
    eye_openness_mean, eye_openness_std = calculate_eye_openness_stats(df)
    pupil_path_length = calculate_pupil_path_length(df)
    result = create_feature_dataframe(eye_openness_mean, eye_openness_std, pupil_path_length)
    print(result)
    plot_3d_scatter_with_bins(x, y, z)
