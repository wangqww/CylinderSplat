import matplotlib.pyplot as plt
import numpy as np

# 外参数据（每张图片的位置信息）
camera_positions = [
    [52.8485802683147, -21.304031700411578, 1.6413706956336251],
    [53.29248777792229, -21.379382495322712, 1.6859902478626596],
    [53.74727024775089, -21.47898088375939, 1.6368908859833404],
    [54.26482063652099, -21.54807039836799, 1.696575657144663],
    [54.71284992338392, -21.595990273577236, 1.6720353125397847],
    [55.157832336460416, -21.660196362022162, 1.7106902173561551],
    [55.6805590906579, -21.754654197592554, 1.6881682822994426],
    [56.139338314537326, -21.800146748532455, 1.6813019844988617],
    [56.58179519356311, -21.859606904118976, 1.7165550846107072],
    [57.07639421974714, -22.023529387566587, 1.6773897263877564]
]

# 将数据转换为 NumPy 数组以便处理
camera_positions = np.array(camera_positions)

# 提取 X, Y, Z 坐标
x = camera_positions[:, 0]
y = camera_positions[:, 1]

# 创建一个 3D 绘图
fig = plt.figure()

# 绘制相机轨迹
plt.plot(x, y, marker='o', color='b', label='Camera Trajectory')

# 添加标签和标题
plt.set_xlabel('X Position')
plt.set_ylabel('Y Position')
plt.set_title('Camera Movement Trajectory')
plt.legend()

# 保存图形到文件
plt.savefig("camera_trajectory.png", dpi=300)  # 设置文件名和分辨率（dpi）

# 显示图形
plt.show()
plt.close()



import torch
import matplotlib.pyplot as plt
import numpy as np

w2w = torch.tensor([  #  X -> X, Z -> Y, upY -> Z
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).float()

# 外参数据（10张图片，每张图片对应一个 4x4 的外参矩阵）
data = torch.inverse(extrinsics_orig[[131]]) @ extrinsics_orig[[131,132,133,134,135,136,137,138,139,140]]
data = w2w @ torch.inverse(extrinsics_orig[[131]]) @ extrinsics_orig[[131,132,133,134,135,136,137,138,139,140]]

# 提取每一张图片的平移向量（最后一列）
camera_positions = data[:, :, 3].numpy()

# 提取 X 和 Y 坐标
x = camera_positions[:, 0]
z = camera_positions[:, 2]

# 创建一个 2D 绘图
plt.figure()

# 绘制相机轨迹
plt.plot(x, z, marker='o', color='b', label='Camera Trajectory')
# 设置 X 和 Y 坐标轴比例一致
plt.axis('equal')
# 添加标签和标题
plt.xlabel('X Position')
plt.ylabel('Z Position')
plt.title('Camera Movement in XZ Plane')
plt.legend()

# 如果需要保存图形，可以使用以下命令
plt.savefig("camera_trajectory_xy3.png", dpi=300)
plt.close()

