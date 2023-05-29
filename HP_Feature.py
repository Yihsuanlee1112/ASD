import math

def distance(x, y, z):
    # 計算三個軸上的移動距離的平方和
    distance_squared = x**2 + y**2 + z**2
    # 取平方根得到真正的移動距離
    distance = math.sqrt(distance_squared)
    return distance

print(distance(21.405594, 12.17435, 28.787631))