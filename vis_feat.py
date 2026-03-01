import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import open3d as o3d
from skimage.measure import block_reduce # 导入块缩减函数
from scipy.ndimage import map_coordinates

def reshape_normalize(x):
    '''
    Args:
        x: [B, C, H, W]

    Returns:

    '''
    B, C, H, W = x.shape
    x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator==0, 1, denominator)
    return x / denominator

def normalize(x):
    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1, denominator)
    return x / denominator

def features_to_blocky_heatmap(
    sat_features,
    idx=0,
    img_name='test_blocky_heatmap.png',
    aggregation_method='pca',
    cmap_name='inferno',
    pixels_per_block=4, # 【核心修改】定义每个块状格子的像素边长
    final_w=256,
    use_polar_transform=False, # 将 Cartesian 参数改为更清晰的名称
    block=False, # 是否分块
):
    """
    将多通道特征图转换为块状（大格子）风格的热力图。
    采用“固定格子大小”逻辑，确保在不同分辨率输入下格子密度均匀。
    
    Args:
        pixels_per_block (int): 每个最终的块状格子代表的原始特征图区域的边长（像素）。
        use_polar_transform (bool): 如果为True，则将输入特征图从极坐标(H=θ, W=r)转为笛卡尔坐标进行可视化。
    """
    # --- 辅助函数 (保持不变) ---
    def reshape_for_pca(feature_map):
        C, H, W = feature_map.shape
        return feature_map.transpose(1, 2, 0).reshape(H * W, C)

    def normalize_0_1(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    # 1. 提取并转换数据
    if hasattr(sat_features, 'data'):
        sat_feat_slice = sat_features[idx, :, :, :].data.cpu().numpy()
    else:
        sat_feat_slice = sat_features[idx, :, :, :]

    # 2. 聚合为单通道强度图
    C, H_orig, W_orig = sat_feat_slice.shape
    if aggregation_method == 'mean':
        intensity_map = np.mean(sat_feat_slice, axis=0)
    # ... (其他聚合方法逻辑不变)
    elif aggregation_method == 'pca':
        reshaped_features = reshape_for_pca(sat_feat_slice)
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(reshaped_features)
        intensity_map = principal_component.reshape(H_orig, W_orig)
    else:
        raise ValueError(f"不支持的聚合方法: {aggregation_method}")

    # 3. （可选）执行极坐标到笛卡尔坐标的转换
    if use_polar_transform:
        # 创建一个与原图相同尺寸的笛卡尔坐标网格
        x = np.linspace(-1, 1, W_orig)
        y = np.linspace(-1, 1, H_orig)
        xv, yv = np.meshgrid(x, y)
        radius = np.sqrt(xv**2 + yv**2)
        theta = np.arctan2(yv, xv)
        
        # 将半径归一化到 [0, 1] 范围内
        max_radius = np.max(radius)
        if max_radius > 0:
            radius /= max_radius
        
        w_coords = radius * (W_orig - 1)
        h_coords = (theta + np.pi) / (2 * np.pi) * (H_orig - 1)
        
        coords = np.stack([h_coords, w_coords])
        final_map = map_coordinates(intensity_map, coords, order=1, cval=0.0)
    else:
        final_map = intensity_map
        
    # 5. 归一化低分辨率强度图并应用颜色映射 (保持不变)
    if block:
        # 4. 【核心步骤】使用“固定格子大小”进行降采样
        H, W = final_map.shape # 获取待处理图的尺寸
        
        # 直接定义块的大小
        block_h = pixels_per_block
        block_w = pixels_per_block
        
        print(f"原始尺寸: ({H_orig}, {W_orig}) -> 待处理尺寸: ({H}, {W})")
        print(f"每个格子的像素大小固定为: ({block_h}, {block_w})")

        if H < block_h or W < block_w:
            raise ValueError(f"pixels_per_block ({pixels_per_block}) 不能大于图片尺寸 ({H}, {W})。")

        # 使用 block_reduce 进行平均池化，得到低分辨率的强度图
        # 最终的格子数会根据图片尺寸和格子大小自动计算得出
        low_res_map = block_reduce(final_map, block_size=(block_h, block_w), func=np.mean)
        low_h, low_w = low_res_map.shape
        normalized_map = normalize_0_1(low_res_map)
        print(f"块状热力图格子数: {low_res_map.shape}")
    else:
        low_h, low_w = final_map.shape
        normalized_map = normalize_0_1(final_map)
    cmap = plt.get_cmap(cmap_name)
    heatmap_rgba = cmap(normalized_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 6. 将低分辨率的彩色图放大，并保持块状效果
    low_res_img = Image.fromarray(heatmap_rgb)
    
    # 放大时，需要保持 low_res_map 的长宽比
    aspect_ratio = low_h / low_w if low_w > 0 else 0
    final_h = int(round(final_w * aspect_ratio))
    
    final_img = low_res_img.resize((final_w, final_h), Image.Resampling.NEAREST)

    # 7. 保存图像
    final_img.save(img_name)
    print(f"块状热力图已保存至: {img_name}")


def visualize_polar_as_cartesian(
    polar_features,
    idx=0,
    img_name='test_polar_to_cartesian.png',
    aggregation_method='pca',
    cmap_name='inferno',
    output_size=512  # 定义最终输出的笛卡爾图像边长
):
    """
    将极坐标下的特征图 (H=theta, W=r) 转换为笛卡尔坐标 (y, x) 并可视化。
    """
    # --- 辅助函数 (与之前类似) ---
    def normalize_0_1(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    # 1. 提取特征并聚合为单通道强度图
    if hasattr(polar_features, 'data'):
        polar_feat_slice = polar_features[idx, :, :, :].data.cpu().numpy()
    else:
        polar_feat_slice = polar_features[idx, :, :, :]

    C, H_theta, W_r = polar_feat_slice.shape

    # 使用与之前相同的方法聚合特征
    if aggregation_method == 'mean':
        intensity_map_polar = np.mean(polar_feat_slice, axis=0)
    elif aggregation_method == 'pca':
        reshaped = polar_feat_slice.transpose(1, 2, 0).reshape(H_theta * W_r, C)
        pca = PCA(n_components=1)
        pc = pca.fit_transform(reshaped)
        intensity_map_polar = pc.reshape(H_theta, W_r)
    else:
        raise ValueError(f"不支持的方法: {aggregation_method}")

    # --- 核心步骤: 极坐标到笛卡尔坐标的逆向变换 ---

    # 2. 创建输出的笛卡尔坐标网格
    # 创建一个 (output_size x output_size) 的网格
    x = np.linspace(-1, 1, output_size)
    y = np.linspace(-1, 1, output_size)
    xv, yv = np.meshgrid(x, y)

    # 3. 将笛卡尔坐标 (x, y) 转换为极坐标 (r, theta)
    # r 的范围是 [0, sqrt(2)]，我们将其归一化到 [0, 1]
    radius = np.sqrt(xv**2 + yv**2)
    # theta 的范围是 [-pi, pi]
    theta = np.arctan2(yv, xv)

    # 4. 将极坐标 (r, theta) 映射到输入特征图的索引 (h, w)
    #   - 半径 r -> 宽度 w (w_r 对应最大半径)
    #     我们将笛卡尔网格中的最大半径1映射到极坐标图的最外圈
    w_coords = radius * (W_r - 1)
    
    #   - 角度 theta -> 高度 h (h_theta 对应 2*pi)
    #     将 [-pi, pi] 的 theta 映射到 [0, H_theta-1] 的 h 索引
    h_coords = (theta + np.pi) / (2 * np.pi) * (H_theta - 1)

    # 5. 使用 map_coordinates 进行插值
    # map_coordinates 需要的坐标格式是 [[h1,h2,...], [w1,w2,...]]
    coords = np.stack([h_coords, w_coords])
    
    # order=1 表示双线性插值，cval=0.0 表示在原始图像范围外的点填充为0（黑色）
    cartesian_map = map_coordinates(intensity_map_polar, coords, order=1, cval=0.0)

    # --- 可视化 ---

    # 6. 归一化并应用颜色映射
    normalized_map = normalize_0_1(cartesian_map)
    cmap = plt.get_cmap(cmap_name)
    heatmap_rgba = cmap(normalized_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    
    # 7. 保存图像
    final_img = Image.fromarray(heatmap_rgb)
    final_img.save(img_name)
    print(f"极坐标特征图已转换为笛卡尔坐标热力图并保存至: {img_name}")


def single_features_to_RGB(sat_features, idx=0, img_name='test_img.png'):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, H, W = sat_feat.shape
    flatten = np.concatenate([sat_feat], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    # sat = sat.resize((512, 512))
    sat.save(img_name)

def reduce_gaussian_features_to_rgb(features):
    """
    使用PCA将高维特征降维至3维，并归一化为RGB值。

    参数:
    features (np.ndarray): 输入的特征数组，形状为 [B, N, C]。
                           例如：[6, 102400, 128]。

    返回:
    np.ndarray: 降维并归一化后的RGB特征，形状为 [B, N, 3]，数值范围在 [0, 1]。
    """
    # 1. 记录原始形状
    features = features.detach().cpu()
    B, N, C = features.shape
    print(f"原始特征形状: {features.shape}")

    # 2. 将数据重塑为2D数组 (B*N, C)，以便PCA处理
    #    PCA是按样本进行分析的，这里我们将B*N个点都看作样本
    features_reshaped = features.reshape(-1, C)
    print(f"重塑后用于PCA的形状: {features_reshaped.shape}")

    # 3. 初始化并执行PCA
    #    n_components=3 表示我们希望将特征降到3维
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)
    print(f"PCA降维后的形状: {features_pca.shape}")

    # 4. 将降维后的数据重塑回原始的批次和空间维度
    features_pca_reshaped = features_pca.reshape(B, N, 3)
    print(f"恢复批次和空间维度后的形状: {features_pca_reshaped.shape}")

    # 5. (关键步骤) 归一化到[0, 1]范围，方便渲染为RGB颜色
    #    PCA的输出范围不是固定的，直接可视化效果会很差
    #    我们对每个通道（主成分）独立进行min-max归一化
    min_vals = features_pca_reshaped.min(axis=(0, 1), keepdims=True)
    max_vals = features_pca_reshaped.max(axis=(0, 1), keepdims=True)
    
    # 防止除以零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    features_rgb = (features_pca_reshaped - min_vals) / range_vals
    print(f"最终RGB特征形状: {features_rgb.shape}")
    print(f"RGB特征的最小值: {features_rgb.min()}, 最大值: {features_rgb.max()}")

    return features_rgb

def point_features_to_rgb_colormap(point_features, cmap_name='viridis', zero_threshold=1e-6):
    """
    将点云的高维特征通过PCA降维并应用Colormap，生成RGB颜色。
    原始特征值都接近于零的点将被设置为黑色。

    参数:
    point_features (torch.Tensor or np.ndarray): 输入的特征张量，形状为 [B, N, C]。
    cmap_name (str): 要使用的matplotlib colormap的名称。
    zero_threshold (float): 用于判断特征值是否接近于零的阈值。

    返回:
    np.ndarray: 降维并应用颜色图后的RGB特征，形状为 [B, N, 3]，数值范围在 [0, 1]。
    """
    # --- 1. 确保数据为NumPy数组并获取形状 ---
    if hasattr(point_features, 'detach'): # 检查是否为PyTorch Tensor
        features = point_features.detach().cpu().numpy()
    elif isinstance(point_features, np.ndarray):
        features = point_features
    else:
        raise TypeError("输入必须是PyTorch张量或NumPy数组")
        
    B, N, C = features.shape
    print(f"原始特征形状: {features.shape}")

    # --- 2. 为全局PCA和Masking重塑数据 ---
    # 将所有批次的所有点合并，以便进行统一的PCA拟合
    features_reshaped = features.reshape(-1, C) # 形状变为 [B*N, C]
    
    # --- 3. 识别“零特征”点 ---
    # 在所有点中，找到那些所有特征通道都接近于零的点
    is_zero_mask_flat = np.all(np.abs(features_reshaped) < zero_threshold, axis=-1) # 形状为 [B*N]
    
    # --- 4. 执行PCA ---
    # 在所有点上拟合PCA，以确保颜色映射的全局一致性
    print("正在对所有点执行PCA...")
    pca = PCA(n_components=1)
    # 使用 fit_transform 学习并转换数据
    pc1_flat = pca.fit_transform(features_reshaped) # 形状为 [B*N, 1]

    # --- 5. 归一化第一主成分到 [0, 1] ---
    # 关键：为了获得更好的对比度，我们仅根据“非零特征”点的范围来确定归一化尺度
    pc1_non_zero = pc1_flat[~is_zero_mask_flat]
    
    if pc1_non_zero.size == 0:
        # 如果所有点都是零特征点，则所有点的颜色值设为0.5（灰色）
        normalized_pc1_flat = np.full_like(pc1_flat, 0.5)
    else:
        min_val = pc1_non_zero.min()
        max_val = pc1_non_zero.max()
        
        if max_val == min_val:
            # 如果所有非零点的值都一样，也设为0.5
            normalized_pc1_flat = np.full_like(pc1_flat, 0.5)
        else:
            # 使用非零点的范围进行归一化
            normalized_pc1_flat = (pc1_flat - min_val) / (max_val - min_val)
    
    # 将超出[0,1]范围的值裁剪掉，这可能发生在零特征点上
    normalized_pc1_flat = np.clip(normalized_pc1_flat, 0.0, 1.0)

    # --- 6. 应用Colormap ---
    print(f"正在应用 '{cmap_name}' 颜色图...")
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"警告: Colormap '{cmap_name}' 不存在，将使用 'viridis'。")
        cmap = plt.get_cmap('viridis')

    # cmap应用于一个1D数组会返回一个 [B*N, 4] 的RGBA数组
    colored_points_flat = cmap(normalized_pc1_flat.flatten())[:, :3] # 我们只取RGB，丢弃Alpha通道

    # --- 7. 应用零值掩码 ---
    # 将原始特征为零的点的颜色设置为黑色 (0, 0, 0)
    colored_points_flat[is_zero_mask_flat] = 0.0
    
    # --- 8. 恢复原始的批次形状 ---
    colored_points_rgb = colored_points_flat.reshape(B, N, 3)
    print(f"处理完成，返回的RGB颜色形状为: {colored_points_rgb.shape}")

    return colored_points_rgb


def save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply"):
    """
    将NumPy格式的坐标和颜色数据保存为PLY点云文件。

    参数:
    points_xyz (np.ndarray): 点的XYZ坐标，形状为 [N, 3]。
    points_rgb (np.ndarray): 点的RGB颜色，形状为 [N, 3]，数值范围应在 [0, 1] 之间。
    filename (str): 要保存的文件名。
    """
    # 1. 创建一个open3d的点云对象
    pcd = o3d.geometry.PointCloud()

    # 2. 将NumPy数组赋值给点云对象的points属性
    #    open3d需要的数据类型是Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    print(f"成功加载 {len(pcd.points)} 个点。")

    # 3. 将NumPy数组赋值给点云对象的colors属性
    #    颜色值的范围必须在 [0, 1] 之间
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    print(f"成功加载 {len(pcd.colors)} 个点的颜色。")

    # 4. 将点云对象写入文件
    #    write_ascii=True可以生成人类可读的文本文件，方便调试
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    print(f"点云已成功保存到当前目录下的 '{filename}' 文件中。")
    print("您可以使用MeshLab, CloudCompare或Blender等软件打开查看。")


# ===================================================================
# 1. 新增的辅助函数，用于将计数值张量可视化为矩形蓝色热力图
# ===================================================================
def visualize_counts_as_heatmap(count_tensor, h, w, filename, cmap_name='Blues'):
    """
    将计数值张量可视化为热力图并保存。

    参数:
    count_tensor (torch.Tensor): 形状为 [H*W, C] 或 [H*W] 的计数值张量。
    h (int): 热力图的高度。
    w (int): 热力图的宽度。
    filename (str): 保存图像的文件名。
    cmap_name (str): Matplotlib Colormap的名称。
    """
    print(f"正在可视化并保存到: {filename} ...")
    
    # a. 将张量移至CPU并转为NumPy
    counts = count_tensor.detach().cpu().numpy()

    # b. 我们只关心计数值，所以取第一个通道或求和即可
    #    这里我们假设所有通道的计数值都一样，取第一个通道
    if counts.ndim > 1:
        counts = counts[:, 0]

    # c. 重塑为2D图像形状 [H, W]
    count_map = counts.reshape(h, w)

    # d. 归一化到 [0, 1] 范围
    # --- 修改开始 ---
    # 使用对数缩放来增强低计数值的对比度
    # 加1是为了避免 log(0) 出现错误
    log_counts = np.log1p(count_map)  # log1p(x) is equivalent to log(1 + x)
    max_log_count = log_counts.max()

    if max_log_count > 0:
        normalized_map = log_counts / max_log_count
    else:
        normalized_map = np.zeros_like(count_map, dtype=np.float32)

    # e. 应用颜色图 (例如 'Blues', 'viridis', 'jet')
    #    cmap函数返回的是 [H, W, 4] 的RGBA图像，数值范围[0,1]
    colored_map = cm.get_cmap(cmap_name)(normalized_map)

    # f. 转换为 [0, 255] 的8位整数，并丢弃Alpha通道
    img_array = (colored_map[:, :, :3] * 255).astype(np.uint8)

    # g. 使用Pillow保存为图片
    Image.fromarray(img_array).save(filename)

# ===================================================================
# 2. 新增的函数，用于将计数值张量可视化为圆形（极坐标）热力图
# ===================================================================
def visualize_counts_as_polar_heatmap(count_tensor, num_r, num_theta, filename, cmap_name='Blues'):
    """
    将计数值张量在极坐标系中可视化为圆形热力图并保存为带透明背景的PNG图片。

    参数:
    count_tensor (torch.Tensor): 形状为 [num_r * num_theta, C] 或 [num_r * num_theta] 的计数值张量。
    num_r (int): 半径方向的网格数 (圆心到边缘的划分数量)。
    num_theta (int): 角度方向的网格数 (圆周的划分数量)。
    filename (str): 保存图像的文件名 (推荐使用.png格式以支持透明度)。
    cmap_name (str): Matplotlib Colormap的名称。
    """
    print(f"正在可视化圆形热力图并保存到: {filename} ...")

    # a. 将张量移至CPU并转为NumPy
    counts = count_tensor.detach().cpu().numpy()
    if counts.ndim > 1:
        counts = counts[:, 0]

    # b. 将一维数据重塑为极坐标网格 [半径, 角度]
    polar_grid = counts.reshape(num_r, num_theta)

    # c. 使用对数缩放来增强对比度
    log_grid = np.log1p(polar_grid)
    max_log_val = log_grid.max()
    normalized_grid = log_grid / max_log_val if max_log_val > 0 else np.zeros_like(log_grid)

    # d. 创建输出图像的笛卡尔坐标网格
    #    图像尺寸设为直径，即 2 * num_r
    img_size = num_r * 2
    x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))

    # e. 将笛卡尔坐标的原点移到图像中心
    x_centered = x - num_r + 0.5
    y_centered = y - num_r + 0.5

    # f. 将每个像素的 (x, y) 坐标转换为极坐标 (r, theta)
    #    计算半径 (距离中心的距离)
    cartesian_r = np.sqrt(x_centered**2 + y_centered**2)
    #    计算角度, np.arctan2 返回 [-pi, pi]
    cartesian_theta = np.arctan2(y_centered, x_centered)

    # g. 将计算出的 (r, theta) 映射到我们的极坐标数据网格的索引
    #    半径索引
    r_idx = cartesian_r.astype(int)
    #    角度索引: 将 [-pi, pi] 映射到 [0, num_theta-1]
    theta_idx = ((cartesian_theta + np.pi) / (2 * np.pi) * num_theta).astype(int)
    #    防止索引越界
    theta_idx = np.clip(theta_idx, 0, num_theta - 1)

    # h. 创建一个带Alpha通道的透明画布 [H, W, 4]
    #    (R, G, B, Alpha), 初始全部透明 (Alpha=0)
    img_rgba = np.zeros((img_size, img_size, 4), dtype=np.uint8)

    # i. 填充颜色：只填充在最大半径内的像素
    #    创建一个布尔掩码，标记所有在圆形内的像素
    valid_mask = r_idx < num_r
    
    # j. 根据索引从归一化网格中获取对应的值
    values_to_colorize = normalized_grid[r_idx[valid_mask], theta_idx[valid_mask]]

    # k. 应用颜色图
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(values_to_colorize)  # 返回 [N, 4] 的 RGBA 图像，数值范围 [0, 1]

    # l. 将颜色值 [0, 1] 转换为 [0, 255] 的8位整数，并应用到画布的有效区域
    img_rgba[valid_mask] = (colors * 255).astype(np.uint8)

    # m. 使用Pillow从RGBA数组创建图像并保存
    Image.fromarray(img_rgba, 'RGBA').save(filename)

def show_vis_points(
    reference_points_cam, 
    idx=0, 
    point_size=50, 
    background_image=None,
    output_filename='sample.png',
    background_alpha=0.5
):
    """
    可视化采样点。可以选择性地将点绘制在一张背景图上，并确保输出比例和方向正确。

    Args:
        reference_points_cam (torch.Tensor): 形状为 [B, N, S, D, 2] 的采样点张量。
        idx (int): 要可视化的深度索引 (D)。
        point_size (int): 散点的大小。
        background_image (torch.Tensor, optional): 形状为 [3, H, W] 的RGB图像张量。默认为 None。
        output_filename (str): 保存图像的文件名。
    """
    # --- 1. 提取采样点坐标 (保持不变) ---
    vis_points = reference_points_cam[0, 0, :, idx, :].view(-1, 2)
    if vis_points.is_cuda:
        points_np = vis_points.cpu().detach().numpy()
    else:
        points_np = vis_points.detach().numpy()

    u_coords = points_np[:, 0]
    v_coords = points_np[:, 1]

    # --- 2. 根據有無背景圖，預先計算好所有繪圖參數 (保持不变) ---
    fig_width_inches = 10
    bg_img_np = None

    if background_image is not None:
        C, H, W = background_image.shape
        aspect_ratio = H / W
        fig_height_inches = fig_width_inches * aspect_ratio
        bg_img_np = background_image.cpu().permute(1, 2, 0).numpy()
    else:
        aspect_ratio = 0.5
        fig_height_inches = fig_width_inches * aspect_ratio

    # --- 3. 使用計算好的尺寸創建畫布 (保持不变) ---
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
    
    # --- 4. 繪製背景 (如果存在) (保持不变) ---
    if bg_img_np is not None:
        ax.imshow(bg_img_np, extent=[0, 1, 1, 0], aspect='auto', alpha=background_alpha)

    # --- 5. 繪製散點圖 (保持不变) ---
    ax.scatter(u_coords, v_coords, s=point_size, marker='.')

    # --- 6. 設置坐標軸範圍和標籤 ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # --- 【核心修正】反轉Y軸 ---
    # 無論是否有背景圖，我們都希望Y軸遵循圖像慣例（0在頂部）
    ax.invert_yaxis()
    
    # 隱藏座標軸上的數字標籤，但保留框架
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0) # 隱藏刻度線

    if bg_img_np is None:
        ax.set_aspect(aspect_ratio, adjustable='box')

    # ax.set_xlabel("u (Normalized Width Coordinate)")
    # ax.set_ylabel("v (Normalized Height Coordinate)")
    # ax.set_title("Visualization of Sampled Points")
    ax.grid(True, linestyle='--', alpha=0.4, color='black') # 網格線改為黑色
    
    plt.tight_layout()

    # --- 7. 保存並關閉圖形 (保持不变) ---
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"可視化圖像已保存至: {output_filename}")