import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# 左眼資料其實是右眼(VIVE)
# 右眼資料其實是左眼(VIVE)

# 讀取 Excel 檔案
df = pd.read_excel(
    "D:\\ASD\\【Vive】特徵計算\\Pre\\ASD001_Pre\\puzzle\\LabRightEyeData_Puzzle.xlsx",
    usecols=["Value.gaze_origin_mm.X", "Value.gaze_origin_mm.Y", "Value.gaze_origin_mm.Z"]
)

# # 繪製三維熱圖
# fig = px.scatter_3d(df, x="Value.gaze_origin_mm.X", y="Value.gaze_origin_mm.Y", z="Value.gaze_origin_mm.Z", color="Value.gaze_origin_mm.Z", opacity=0.7, color_continuous_scale="viridis")

# # 顯示圖片
# fig.show()

# 繪製密度圖
sns.kdeplot(data=df, x="Value.gaze_origin_mm.X", y="Value.gaze_origin_mm.Y", 
            # cmap="Spectral_r", 
            cmap="rainbow",
            fill=True, 
            cbar=True, 
            cbar_kws={'ticks': [0.05, 0, 0.5, 1, 2]})
            # cbar_kws={'ticks': [0.01, 0.05, 0.1, 0.15,0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 1]})
plt.title('Left Eye Gaze Map (Puzzle)')

# LabRightEyeData設定 X 軸和 Y 軸範圍
# plt.xlim(-35, -20)
# plt.ylim(-5, 10)

# LabLeftEyeData設定 X 軸和 Y 軸範圍
# plt.xlim(20, 36)
# plt.ylim(-5, 10)

# 顯示圖片
plt.show()