import numpy as np
import glob
import os
import sys
import skimage.io as io
from scipy import ndimage
import scipy.io as sio
import cv2
import scipy
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -----------Train------------
# 配置变量
in_root_dir = 'D:\\GraduateWork\\code\\1\\Train12.8'
out_root_dir = 'D:\\GraduateWork\\code\\1\\Train12.8'


# ------------Test------------
# # 配置变量
# in_root_dir = 'D:\\GraduateWork\\code\\1\\Test'
# out_root_dir = 'D:\\GraduateWork\\code\\1\\CoNSeP_test'


# 定义细胞类别
classes_max_indx = 4
'''
	1 =  CD8+T-bet+ (sky blue)
	2 =  CD8+T-bet- (red)
    3 = CK+PD-L1+ (green)
    4 = CK+PD-L1-(yellow)
'''
color_set = {1: (0, 162, 232), 2: (255, 0, 0), 3: (0, 255, 0),4: (255, 255, 0)}

# 定义输入与输出类别映射关系
class_group_mapping_dict = {1:[0],2:[1],3:[2],4:[3]}  # 类别映射字典，将原始类别映射到新的类别
n_grouped_class_channels = 5  # 新的类别通道数

# 定义图像缩放比例
# img_scale = 1.0

# 如果为True，将移除在5像素距离内标记的重复细胞
remove_duplicates = False


"""-
    此代码假设输入具有以下格式：
        - 在 <in_root_dir> 下：
            Images文件夹：带有该幻灯片的标记图像块
            Labels文件夹：包含图像中每个图像块的标签的mat文件
        - mat文件具有以下变量：
            inst_centroid：形状为 n x 2 的数组，其中 n 是细胞数，坐标为 (x, y)
            inst_type：一个数组，保存着每个细胞在inst_centroid中的细胞类别。类别类型是从1开始的连续整数。
                       这就是为什么color_set字典的键从1开始，表示细胞类别。
                       我们的数据是0,1,2,3，但是在class_grouped_class_channels做了映射

"""


def gaussian_filter_density(img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    """
        从点构建KD树，并为每个点获取最近的邻居点。
        默认的高斯宽度是9。
        Sigma自适应选择为min(最近邻距离*0.125, 2)，并截断为2*sigma。
        在生成每个点的高斯后，将其归一化并添加到最终的密度图中。
        生成的地图的可视化保存在 <slide_name>_<img_name>.png 和 <slide_name>_<img_name>_binary.png 中
    """
    overlap = 'cover'  # 重叠区域选择什么处理方法，add or cover
    # 获取输入图像的形状
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")

    # 创建一个与输入图像相同形状的零矩阵来存储密度估计
    density = np.zeros(img_shape, dtype=np.float32)

    # 创建一个与输入图像形状和类别映射相同形状的零矩阵来存储每个类别的密度估计
    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)

    # 如果没有提供结束坐标，将其设置为图像的边界
    if (end_y <= 0):
        end_y = img.shape[0]
    if (end_x <= 0):
        end_x = img.shape[1]

    # 获取点的数量
    gt_count = len(points)

    # 如果没有点，返回空的密度图
    if gt_count == 0:
        return density

    # 构建KD树以便于快速查找最近的邻居
    leafsize = 2048
    # 构建KD树
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)

    # 查询KD树以获取每个点的最近邻
    distances, locations = tree.query(points, k=2)
    print('generate density...')

    # 最大高斯核的标准差
    max_sigma = 4

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)

        # 检查点是否在指定的图像坐标范围内
        if (pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue

        # 将点的坐标转换为相对于图像起始位置的坐标
        pt[1] -= start_y
        pt[0] -= start_x

        # 如果点在图像内部，将相应位置设置为1
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue

        # 根据最近邻的距离自适应选择高斯核的标准差(Sigma)
        if gt_count > 1:
            sigma = (distances[i][1]) * 0.18
            sigma = min(max_sigma, sigma)
        else:
            sigma = max_sigma

        # 计算高斯核的大小(kernel_size)和标准差(sigma)
        kernel_size = min(max_sigma * 4, int(4 * sigma + 2.0))
        sigma = kernel_size / 2.7
        kernel_width = kernel_size * 2 + 5
        # if(kernel_width < 9):
        #    print('i',i)
        #    print('distances',distances.shape)
        #    print('kernel_width',kernel_width)

        # 使用高斯滤波器生成点的密度估计
        pnt_density = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant', truncate=2)

        # # 将高斯掩码缩放到[0,1]
        # pnt_density = (pnt_density - np.min(pnt_density)) / (np.max(pnt_density) - np.min(pnt_density))

        # 归一化高斯掩码，确保最大值为1（高斯掩码是所有值加起来为1，而不是最大值为1）
        pnt_density = pnt_density / np.max(pnt_density)

        density += pnt_density
        # 获取点的类别
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()
        # 更新相应类别的密度图
        density_class[:, :, class_indx] += pnt_density


    #density_class.astype(np.float16).dump(out_filepath)
    #density.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_all.npy')

    if overlap == 'cover':  # 对重叠区域处理方式选择 cover，即覆盖则限制数值为0-1
        # np.clip()将数组中的值限制在某个范围内
        density = np.clip(density, 0, 1)
        density_class = np.clip(density_class, 0, 1)
    print("Density min:", density.min(), "Density max:", density.max())
    print("Density_class min:", density_class.min(), "Density_class max:", density_class.max())

    # 将密度图和类别密度图保存为二进制文件
    # (density_class > 0).astype(np.uint8).dump(out_filepath)
    # (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    # 将密度图和类别密度图保存为float类型文件
    np.save(out_filepath, density_class.astype(np.float32))
    np.save(os.path.splitext(out_filepath)[0] + '_all.npy', density.astype(np.float32))

    io.imsave(out_filepath.replace('.npy', '.png'), (density * 255).astype(np.uint8))
    # 将密度图保存为二进制图像
    io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
    # 保存每个类别的二进制密度图
    for s in range(1, density_class.shape[-1]):
        io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                  ((density_class[:, :, s] > 0) * 255).astype(np.uint8),
                  vmin=0, vmax=255)


    '''
        9.18更新日志：
            在该方法中输入的img参数是检测点图，是已经带有二进制点的图像，下面的方法可以保证输出原始密度图像，而不是二进制密度图像。
            但是会被检测点的二进制覆盖住，所以要在该方法外面进行操作，来将原始密度图像叠加到原始图像上。
            尝试去利用该方法的输出，应该可行。
        
        修改代码：
            # 将原始密度图像叠加到原始图像上（用以观察高斯掩码是否正确覆盖了细胞）
            img_with_density = img.copy()
            density_normalized = (density / density.max() * 255).astype(np.uint8)
            img_with_density[:, :, 1] = density_normalized
            # 保存叠加后的图像为PNG格式
            img_with_density = img_with_density.astype(np.uint8)
            io.imsave(out_filepath.replace('.npy', '_overlay.png'), img_with_density)
    '''

    print('done.')

    # 返回密度图和类别密度图
    return density.astype(np.float32), density_class.astype(np.float32)

def generate_circle_mask(img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    """
        生成圆形掩码和每个类的密度图，与输入图像大小和点的类别映射相同大小
    """
    overlap = 'cover'  # 重叠区域选择什么处理方法，add or cover
    # 获取输入图像的形状
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "circle mask.")

    # 创建一个与输入图像相同形状的零矩阵来存储密度估计
    density = np.zeros(img_shape, dtype=np.float32)

    # 创建一个与输入图像形状和类别映射相同形状的零矩阵来存储每个类别的密度估计
    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)

    # 如果没有提供结束坐标，将其设置为图像的边界
    if end_y <= 0:
        end_y = img.shape[0]
    if end_x <= 0:
        end_x = img.shape[1]

    # 获取点的数量
    gt_count = len(points)

    # 如果没有点，返回空的密度图
    if gt_count == 0:
        return density

    # 构建KD树以便于快速查找最近的邻居
    leafsize = 2048
    # 构建KD树
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)

    # 查询KD树以获取每个点的最近邻
    distances, locations = tree.query(points, k=2)
    print('generate density...')

    # 最大圆半径
    max_radius = 13

    for i,pt in enumerate(points):
        circle_mask = np.zeros(img_shape, dtype=np.float32)

        # 检查点是否在指定的图像坐标范围内
        if pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x:
            continue

        # 将点的坐标转换为相对于图像起始位置的坐标
        pt[1] -= start_y
        pt[0] -= start_x

        # 根据最近邻的距离自适应选择圆的半径
        if gt_count > 1:
            radius = int((distances[i][1]) * 0.65)
            radius = min(max_radius, radius)
        else:
            radius = max_radius

        # 生成圆形掩码
        cv2.circle(circle_mask, (int(pt[0]), int(pt[1])), radius, 1.0, -1)

        density += circle_mask
        # 获取点的类别
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()  # 1,2,3,4
        # 更新相应类别的密度图
        density_class[:, :, class_indx] += circle_mask

    if overlap == 'cover':  # 对重叠区域处理方式选择 cover，即覆盖则限制数值为0-1
        # np.clip()将数组中的值限制在某个范围内
        density = np.clip(density, 0, 1)
        density_class = np.clip(density_class, 0, 1)
    print("Density min:", density.min(), "Density max:", density.max())
    print("Density_class min:", density_class.min(), "Density_class max:", density_class.max())

    # 将密度图和类别密度图保存为二进制文件
    # (density_class > 0).astype(np.uint8).dump(out_filepath)
    # (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    # 将密度图和类别密度图保存为float类型文件
    np.save(out_filepath, density_class.astype(np.float32))
    np.save(os.path.splitext(out_filepath)[0] + '_all.npy', density.astype(np.float32))

    io.imsave(out_filepath.replace('.npy', '.png'), (density * 255).astype(np.uint8))
    # 将密度图保存为二进制图像
    io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
    # 保存每个类别的二进制密度图
    for s in range(1, density_class.shape[-1]):
        io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                  ((density_class[:, :, s] > 0) * 255).astype(np.uint8),
                  vmin=0, vmax=255)

    print('done.')

    # 返回密度图和类别密度图
    return density.astype(np.float32), density_class.astype(np.float32)

def generate_mix_mask(img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    """
        生成混合掩码，1,2类使用高斯掩码，3,4类使用圆形掩码
    """
    overlap = 'cover'  # 重叠区域选择什么处理方法，add or cover

    # 获取输入图像的形状
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "mix mask.")

    # 创建一个与输入图像相同形状的零矩阵来存储密度估计
    density = np.zeros(img_shape, dtype=np.float32)

    # 创建一个与输入图像形状和类别映射相同形状的零矩阵来存储每个类别的密度估计
    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)

    # 如果没有提供结束坐标，将其设置为图像的边界
    if end_y <= 0:
        end_y = img.shape[0]
    if end_x <= 0:
        end_x = img.shape[1]

    # 获取点的数量
    gt_count = len(points)

    # 如果没有点，返回空的密度图
    if gt_count == 0:
        return density

    # 构建KD树以便于快速查找最近的邻居
    leafsize = 2048
    # 构建KD树
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)

    # 查询KD树以获取每个点的最近邻
    distances, locations = tree.query(points, k=2)
    print('generate density...')

    # 最大圆半径
    max_radius = 13
    # 最大高斯核的标准差
    max_sigma = 4

    for i,pt in enumerate(points):
        mix_mask = np.zeros(img_shape, dtype=np.float32)

        # 检查点是否在指定的图像坐标范围内
        if pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x:
            continue

        # 将点的坐标转换为相对于图像起始位置的坐标
        pt[1] -= start_y
        pt[0] -= start_x

        # 获取点的类别
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()  # 1,2,3,4

        # 根据最近邻的距离自适应选择圆的半径
        if gt_count > 1:
            if class_indx == 1 or class_indx == 2:
                sigma = (distances[i][1]) * 0.18
                sigma = min(max_sigma, sigma)
                # print("正在生成高斯掩码！！！！！！！")
            elif class_indx == 3 or class_indx == 4:
                radius = int((distances[i][1]) * 0.65)
                radius = min(max_radius, radius)
                # print("正在生成圆形掩码！！！！！！！")
        else:
            radius = max_radius
            sigma = max_sigma

        if class_indx == 1 or class_indx == 2:  # 使用高斯掩码
            # 如果点在图像内部，将相应位置设置为1
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                mix_mask[int(pt[1]), int(pt[0])] = 1.
            else:
                continue

            # 计算高斯核的大小(kernel_size)和标准差(sigma)
            kernel_size = min(max_sigma * 4, int(4 * sigma + 2.0))
            sigma = kernel_size / 2.7
            kernel_width = kernel_size * 2 + 5
            # print("kernel_width:", kernel_width)

            # 使用高斯滤波器生成点的密度估计
            mix_mask = scipy.ndimage.filters.gaussian_filter(mix_mask, sigma, mode='constant', truncate=2)

            # 归一化高斯掩码，确保最大值为1（高斯函数是所有值加起来为1，而不是最大值为1）
            mix_mask = mix_mask / np.max(mix_mask)

            # decimal_places = -int(np.floor(np.log10(np.finfo(mix_mask.dtype).eps)))  # 查看有几位小数，目前是7位
            # print(decimal_places)

            # print("max is:", mix_mask.max())
            # print("i", i, ".高斯掩码！！！！")
            #
            # mean_value = np.mean(mix_mask)
            # std_dev = np.std(mix_mask)
            # min_value = np.min(mix_mask)
            # max_value = np.max(mix_mask)
            #
            # print(f"Mean: {mean_value}, Std Dev: {std_dev}, Min: {min_value}, Max: {max_value}")

            # print("Sigma:", sigma)
        elif class_indx == 3 or class_indx == 4:  # 使用圆形掩码
            # 生成圆形掩码
            cv2.circle(mix_mask, (int(pt[0]), int(pt[1])), radius, 1.0, -1)
            # print("max is:", mix_mask.max())
            # print("i", i, ".圆形掩码！！！！")

        # 归一化添加到最终的密度图中(舍弃，这种方法会导致数值过小)
        # mix_mask /= mix_mask.sum()

        density += mix_mask
        # print("Mix mask min:", mix_mask.min(), "Mix mask max:", mix_mask.max())
        # print("Density min:", density.min(), "Density max:", density.max())

        # 更新相应类别的密度图
        density_class[:, :, class_indx] += mix_mask

    if overlap == 'cover':  # 对重叠区域处理方式选择 cover，即覆盖则限制数值为0-1
        # np.clip()将数组中的值限制在某个范围内
        density = np.clip(density, 0, 1)
        density_class = np.clip(density_class, 0, 1)
    print("Density min:", density.min(), "Density max:", density.max())
    print("Density_class min:", density_class.min(), "Density_class max:", density_class.max())

    # 将密度图和类别密度图保存为二进制文件
    # (density_class > 0).astype(np.uint8).dump(out_filepath)
    # (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    # 将密度图和类别密度图保存为float类型文件
    np.save(out_filepath, density_class.astype(np.float32))
    np.save(os.path.splitext(out_filepath)[0] + '_all.npy', density.astype(np.float32))

    io.imsave(out_filepath.replace('.npy', '.png'), (density * 255).astype(np.uint8))
    # 将密度图保存为二进制图像
    io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
    # 保存每个类别的二进制密度图
    for s in range(1, density_class.shape[-1]):
        io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                  ((density_class[:, :, s] > 0) * 255).astype(np.uint8),
                  vmin=0, vmax=255)

    print('done.')

    # 返回密度图和类别密度图
    return density.astype(np.float32), density_class.astype(np.float32)


if __name__ == "__main__":

    '''
        对于每个图像：
            重新缩放修补程序图像和标记的细胞中心的坐标。
                将缩放图像保存为(<out_img_dir>/<img_name>.png)
                并创建具有覆盖在修补程序图像上的不同颜色的细胞类的可视化(保存为<out_gt_dir>/<img_name>_img_with_dots.jpg)
            创建分类点注释映射(保存为<out_gt_dir>/<img_name>_gt_dots.npy)
            创建检测点注释映射(保存为<out_gt_dir>/<img_name>_gt_dots_all.npy)
            生成二进制掩码，其中在每个细胞中心创建一个高斯掩码。高斯的宽度是自适应的，使得细胞不相交。
            通过将所有像素> 0 到 1，其余像素设置为0，将高斯映射保存为二进制掩码。
                分类映射文件保存为<out_gt_dir>/<img_name>.npy
                    和二进制掩码的可视化保存为(<out_gt_dir>/<img_name>_s<class_indx>_binary.png)
                检测映射文件保存为<out_gt_dir>/<img_name>_all.npy
                    和二进制掩码的可视化保存为(<out_gt_dir>/<img_name>_binary.png)
    '''

    '''
        每个 .mat标签文件都有键:
        'inst_type'
        'inst_centroid'
    '''
    in_img_dir = os.path.join(in_root_dir, 'images')  # 输入图像文件夹路径
    in_label_dir = os.path.join(in_root_dir, 'Labels')  # 输入标签文件夹路径

    out_img_dir = os.path.join(out_root_dir, 'images')  # 输出图像文件夹路径
    out_gt_dir = os.path.join(out_root_dir, 'gt_custom')  # 输出自定义标签文件夹路径

    mask = "mix"  # gaussian circle mix

    if not os.path.exists(out_root_dir):
        os.mkdir(out_root_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    if not os.path.exists(out_gt_dir):
        os.mkdir(out_gt_dir)

    img_files = glob.glob(os.path.join(in_img_dir, '*.png'))  # 获取图像文件列表

    for img_filepath in img_files:
        print('img_filepath', img_filepath)

        # 读取图像
        img_name = os.path.splitext(os.path.basename(img_filepath))[0]
        out_gt_dmap_filepath = os.path.join(out_gt_dir, img_name  + '.npy')
        img = io.imread(img_filepath)[:, :, 0:3]

        # 读取.mat文件
        mat_filepath = os.path.join(in_label_dir, img_name + '.mat')
        mat = sio.loadmat(mat_filepath)

        # 读取并缩放细胞中心坐标
        # centroids = (mat["inst_centroid"] * img_scale).astype(int)
        # 读取细胞中心坐标
        centroids = mat["inst_centroid"].astype(int)
        class_types = mat["inst_type"].squeeze()

        # 缩放图像
        # img2 = cv2.resize(img, (int(img.shape[1] * img_scale + 0.5), int(img.shape[0] * img_scale + 0.5)))
        img2 = img.copy()
        img3 = img2.copy()
        # io.imsave(os.path.join(out_img_dir, img_name+'.png'), img2)

        # 初始化标签数组
        patch_label_arr_dots = np.zeros((img2.shape[0], img2.shape[1], classes_max_indx), dtype=np.uint8)

        # 确保缩放图像后坐标仍在范围内
        # print('centroids',centroids.shape)
        # print('class_types',class_types.shape)
        # centroids[(np.where(centroids[:, 1] >= img2.shape[0]), 1)] = img2.shape[0] - 1
        # centroids[(np.where(centroids[:, 0] >= img2.shape[1]), 0)] = img2.shape[1] - 1

        # 生成分类点注释地图
        for dot_class in range(0, classes_max_indx):
            patch_label_arr = np.zeros((img2.shape[0], img2.shape[1]))
            patch_label_arr[(centroids[np.where(class_types == dot_class)][:, 1],
                                centroids[np.where(class_types == dot_class)][:, 0])] = 1
            patch_label_arr_dots[:, :, dot_class] = patch_label_arr
            #patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            # img2[np.where(patch_label_arr > 0)] = color_set[dot_class]

        # 映射class_group_mapping_dict中键的索引到值中的所有类别
        patch_label_arr_dots_grouped = np.zeros((img2.shape[0], img2.shape[1], n_grouped_class_channels), dtype=np.uint8)
        for class_id, map_class_lst in class_group_mapping_dict.items():
            patch_label_arr = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            img3[np.where(patch_label_arr > 0)] = color_set[class_id]
            patch_label_arr_dots_grouped[:, :, class_id] = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
        patch_label_arr_dots = patch_label_arr_dots_grouped

        # 移除重复的点
        if (remove_duplicates):
            for c in range(patch_label_arr_dots.shape[-1]):
                tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant', cval=0.0)
                duplicate_points = np.where(tmp > 1)
                while (len(duplicate_points[0]) > 0):
                    y = duplicate_points[0][0]
                    x = duplicate_points[1][0]
                    patch_label_arr_dots[max(0, y - 2):min(patch_label_arr_dots.shape[0] - 1, y + 3),
                    max(0, x - 2):min(patch_label_arr_dots.shape[1] - 1, x + 3), c] = 0
                    patch_label_arr_dots[y, x, c] = 1
                    tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant',
                                            cval=0.0)
                    duplicate_points = np.where(tmp > 1)

        # 生成分类点标注地图
        patch_label_arr_dots_all = patch_label_arr_dots[:, :, :].sum(axis=-1)
        # Save Dot maps
        patch_label_arr_dots.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots.npy'))
        patch_label_arr_dots_all.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots_all.npy'))

        # 为点标注创建分类图和检测图
        for dot_class in range(1, patch_label_arr_dots.shape[-1]):
            print('dot_class', dot_class)
            print('patch_label_arr_dots[:,:,dot_class]', patch_label_arr_dots[:, :, dot_class].sum())
            patch_label_arr = patch_label_arr_dots[:, :, dot_class].astype(int)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((6, 6)), mode='constant', cval=0.0)
            img2[np.where(patch_label_arr > 0)] = color_set[dot_class]
        io.imsave(os.path.join(out_gt_dir, img_name + '_img_with_dots.jpg'), img2)

        # 生成高斯地图
        # 不要为每个类分别执行此操作，因为这可能会导致检测图中的交叉点
        mat_s_points = np.where(patch_label_arr_dots > 0)
        points = np.zeros((len(mat_s_points[0]), 2))
        print(points.shape)
        points[:, 0] = mat_s_points[1]
        points[:, 1] = mat_s_points[0]

        if mask == "gaussian":
            patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = gaussian_filter_density(
                img2, points, patch_label_arr_dots, out_gt_dmap_filepath
            )
        elif mask == "circle":
            patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = generate_circle_mask(
                img2, points, patch_label_arr_dots, out_gt_dmap_filepath
            )
        elif mask == "mix":
            patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = generate_mix_mask(
                img2, points, patch_label_arr_dots, out_gt_dmap_filepath
            )
        else:
            print("mask is false!")

        # 将密度图像添加到原始图像上
        # 注意：这会导致原始图像信息被修改，该步骤仅用于观察掩码是否完全覆盖，仅此而已
        alpha = 1  # 调整叠加的透明度
        density_map = cv2.imread(out_gt_dmap_filepath.replace('.npy', '.png'))
        composite_img = cv2.addWeighted(img, 1.0, density_map, alpha, 0)
        # 保存叠加后的图像
        cv2.imwrite(out_gt_dmap_filepath.replace('.npy', '_overlay.png'), composite_img)

        # 将二进制密度图像添加到原始图像上
        # 注意：这会导致原始图像信息被修改，该步骤仅用于观察掩码是否完全覆盖，仅此而已
        alpha = 1  # 调整叠加的透明度
        density_map_binary = cv2.imread(out_gt_dmap_filepath.replace('.npy', '_binary.png'))
        composite_img_binary = cv2.addWeighted(img, 1.0, density_map_binary, alpha, 0)
        # 保存叠加后的图像
        cv2.imwrite(out_gt_dmap_filepath.replace('.npy', '_overlay_binary.png'), composite_img_binary)

