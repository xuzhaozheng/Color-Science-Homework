import pandas as pd
import numpy as np
import colour.plotting
import matplotlib.pyplot as plt

# 获取380nm-780nm的光谱分布数据
spectrum_data = pd.read_csv('PhotoResearch_Raw_Data.csv', sep=',').iloc[0:1, 3:404].T

# 获取380nm-780nm的响应值
cmf_xyz_1931 = pd.read_csv('ciexyz31_1nm.csv', header=None, sep=',').iloc[20:421, 1:4].T

# 设定系数k使Y值最终为亮度
k = 683
XYZ_1931_tri = k * np.dot(cmf_xyz_1931.values,spectrum_data.values)  # 使用矩阵乘法完成运算
X, Y, Z = float(XYZ_1931_tri[0]), float(XYZ_1931_tri[1]), float(XYZ_1931_tri[2])

# 计算xy坐标
x = X / (X + Y + Z)
y = Y / (X + Y + Z)

# 计算u' v'坐标
u_ = 4 * X / (X + 15 * Y + 3 * Z)
v_ = 9 * Y / (X + 15 * Y + 3 * Z)


# 计算CCT
n = (x - 0.3320) / (0.1858 - y)
CCT = 437 * pow(n, 3) + 3601 * pow(n, 2) + 6861 * n + 5517
print(CCT)

xy_coord = '{:.04f},{:.04f}'.format(x, y)
u_v_coord = '{:.04f},{:.04f}'.format(u_, v_)

colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
plt.plot(x, y, 'o', markersize=4, color=(0, 0.5, 0))
plt.text(x + 0.01, y + 0.01, xy_coord, fontsize=12, color=(0, 0.5, 0), style="italic")

colour.plotting.plot_chromaticity_diagram_CIE1976UCS(standalone=False)
plt.plot(u_, v_, 'o', markersize=4, color=(0, 0.5, 0))
plt.text(u_ + 0.01, v_ + 0.01, u_v_coord, fontsize=12, color=(0, 0.5, 0), style="italic")
plt.show()
