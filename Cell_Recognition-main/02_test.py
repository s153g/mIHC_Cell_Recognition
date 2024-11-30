import os
import numpy as np
from skimage import io
import cv2
import sys
from skimage.measure import label, moments
from skimage import filters
from skimage.feature import peak_local_max
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
import glob

from model_arch import UnetVggMultihead
from my_dataloader import CellsDataset

checkpoints_root_dir = '../MCSpatNet_checkpoints'  # 存放所有训练输出的根目录
# checkpoints_folder_name = 'float_mix_12.4'  # 当前训练输出文件夹的名称
checkpoints_folder_name = 'float_mix_attention_loss_MSE_binary_lamda_1_bug'  # 当前训练输出文件夹的名称
eval_root_dir = '../MCSpatNet_eval'  # 存放评估结果的根目录
# 该文件只是针对某一测试epoch进行测试验证以及可视化
epoch = 192  # 要测试的模型的训练周期
visualize = True  # 是否输出预测可视化结果
test_data_root = 'D:\\GraduateWork\\code\\1\\Test'  # 测试数据的根目录
# test_data_root = 'D:\\GraduateWork\\code\\1\\Train12.8'  # 测试数据的根目录
test_split_filepath = None

if __name__ == "__main__":

    # 初始化
    # CD8+T-bet+：红、CD8+T-bet-: 青、CK+PD-L1+: 橙、CK+PD-L1-: 黄
    color_set = {0: (255, 0, 0), 1: (0, 255, 255), 2: (255, 165, 0), 3: (255, 255, 0)}

    # 模型检查点和输出配置参数
    models_root_dir = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    out_dir = os.path.join(eval_root_dir, checkpoints_folder_name + f'_e{epoch}')

    if not os.path.exists(eval_root_dir):
        os.mkdir(eval_root_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 数据配置参数
    test_image_root = os.path.join(test_data_root, 'images')  # 测试图像的路径
    test_dmap_root = os.path.join(test_data_root, 'gt_custom')  # 测试图像的密度图的路径
    test_dots_root = os.path.join(test_data_root, 'gt_custom')  # 测试图像的点标注的路径
    test_dmap_all_root = os.path.join(test_data_root, 'gt_custom')  # 验证目标密度图的根目录（用于细胞检测）

    # 模型配置参数
    gt_multiplier = 1
    gpu_or_cpu = 'cuda'  # use cuda or cpu
    dropout_prob = 0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 4  # 类别数
    n_classes_out = n_classes + 1
    class_indx = '1,2,3,4'
    class_weights = np.array([1, 1, 1, 1])

    # ----------------
    # n_clusters = 5
    # n_classes2 = n_clusters * (n_classes)
    #
    # r_step = 15
    # r_range = range(0, 100, r_step)
    # r_arr = np.array([*r_range])
    # r_classes = len(r_range)
    # r_classes_all = r_classes * (n_classes)

    # 下面的变量用于设置阈值的参数
    thresh_low = 0.5  # 阈值下限。用于将概率图转换为二进制图像的下阈值
    thresh_high = 0.5  # 阈值上限。用于将概率图转换为二进制图像的上阈值
    size_thresh = 5  # 尺寸阈值

    # 创建模型实例并移动到指定设备
    device = torch.device(gpu_or_cpu)
    model = UnetVggMultihead(
        kwargs={'dropout_prob': dropout_prob,
                'initial_pad': initial_pad,
                'interpolate': interpolate,
                'conv_init': conv_init,
                'n_classes': n_classes,
                'n_channels': 3,
                'n_heads': 2,
                'head_classes': [1, n_classes]
                }
    )
    model.to(device)
    criterion_sig = nn.Sigmoid()  # 初始化Sigmoid层
    criterion_softmax = nn.Softmax(dim=1)  # 初始化Softmax层

    # 创建测试数据集和数据加载器
    test_dataset = CellsDataset(test_image_root,
                                test_dmap_root,
                                test_dots_root,
                                test_dmap_all_root,
                                class_indx,
                                split_filepath=test_split_filepath,
                                phase='test',
                                fixed_size=-1,
                                max_scale=16
                                )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False
                                              )

    print('thresh', thresh_low, thresh_high)

    # 加载模型
    print('test epoch ' + str(epoch))
    model_files = glob.glob(os.path.join(models_root_dir, 'mcspat_epoch_' + str(epoch) + '_*.pth'))
    model_files2 = glob.glob(os.path.join(models_root_dir, '*epoch_' + str(epoch) + '_*.pth'))
    if ((model_files == None) or (len(model_files) == 0)):
        if ((model_files2 == None) or (len(model_files2) == 0)):
            print('not found ', 'mcspat_epoch_' + str(epoch))
            exit()
        else:
            model_param_path = model_files2[0]
    else:
        model_param_path = model_files[0]

    sys.stdout.flush()
    model.load_state_dict(torch.load(model_param_path), strict=True)
    model.to(device)
    model.eval()

    # 使用torch.no_grad()，避免梯度计算，提高性能
    with torch.no_grad():
        for i, (img, gt_dmap, gt_dots, gt_dmap_all, img_name) in enumerate(tqdm(test_loader, disable=True)):
            img_name = img_name[0]
            # 每个图像都建立一个文件夹，这样方便观察
            img_dir = os.path.join(out_dir, img_name.replace('.png', ''))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            sys.stdout.flush()

            # 前向传播
            img = img.to(device)
            et_dmap_lst, attention_coefficients = model(img)
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]  # 细胞检测预测
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]  # 细胞分类预测
            # 最后一层的注意力系数矩阵，并进行裁剪，上下左右各减少4个像素，以保证和img大小一致(经过Sigmoid的)
            attention_coefficient_last = attention_coefficients[3][:, :, 4:-4, 4:-4]

            # -----------------
            # et_dmap_subclasses = et_dmap_lst[2][:, :, 2:-2, 2:-2]
            # et_kmap = et_dmap_lst[3][:, :, 2:-2, 2:-2] ** 2

            # 处理模型输出和真实标签
            # gt_dmap = gt_dmap > 0
            # gt_dmap_all = gt_dmap.max(1)[0].detach().cpu().numpy()
            gt_dots_all = gt_dots.max(1)[0].detach().cpu().numpy().squeeze()
            gt_dots = gt_dots.detach().cpu().numpy()

            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255
            img_centers_all = img.copy()
            img_centers_all_gt = img.copy()

            img_centers_all_all = img.copy()
            img_centers_all_all_gt = img.copy()

            # 开始：评估所有的检测结果
            g_count = gt_dots_all.sum()

            cv2.imwrite(os.path.join(img_dir, img_name.replace('.png', '_et_all_sig.png')), et_all_sig.squeeze() * 255)
            # 获取预测中的连通组件并应用小尺寸阈值
            e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)
            comp_mask = label(e_hard2)
            e_count = comp_mask.max()
            s_count = 0
            if (size_thresh > 0):
                for c in range(1, comp_mask.max() + 1):
                    s = (comp_mask == c).sum()
                    if (s < size_thresh):
                        e_count -= 1
                        s_count += 1
                        e_hard2[comp_mask == c] = 0
            e_hard2_all = e_hard2.copy()
            # 对小尺寸阈值处理后的图像进行可视化
            cv2.imwrite(os.path.join(img_dir, img_name.replace('.png', '_thresh.png')), e_hard2 * 255)

            # 进行距离变换，并可视化结果
            e_hard2 = cv2.distanceTransform(e_hard2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            # 将距离变换结果归一化到[0,1]
            cv2.normalize(e_hard2, e_hard2, 0, 1, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(img_dir, img_name.replace('.png', '_distance_transform.png')), e_hard2*255)

            # 在局部极大值之前增加20像素
            e_hard2_padded = np.pad(e_hard2, 20, mode='constant', constant_values=0)

            # 获取局部极大值的坐标（min_distance：用于控制局部极大值之间的最小距离的参数）
            local_max_coords = peak_local_max(e_hard2_padded, indices='rc', min_distance=12)

            # 新建一个与原始输入大小相同的数组
            markers = np.zeros_like(e_hard2_padded, dtype=np.uint8)

            # 根据局部极大值坐标生成小圆
            for x, y in local_max_coords:
                cv2.circle(markers, (y, x), 6, 1, -1)

            # 将根据局部极大值生成的数组赋予e_hard2，这样不会影响后续的操作
            # 并裁剪回原始大小
            e_hard2 = markers[20:e_hard2.shape[0]+20, 20:e_hard2.shape[1]+20]

            # 将标记局部极大值后的图像保存
            cv2.imwrite(os.path.join(img_dir, img_name.replace('.png', '_local_maxima.png')),
                        e_hard2 * 255)
            e_hard2 = (e_hard2 > 0).astype(np.uint8)

            # 获取预测中的连通组件的中心点
            e_dot = np.zeros((img.shape[0], img.shape[1]))
            e_dot_vis = np.zeros((img.shape[0], img.shape[1]))
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if (M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot_vis[cy - 1:cy + 1, cx - 1:cx + 1] = 1
                e_dot[min(cy, e_dot.shape[0] - 1), min(cx, e_dot.shape[1] - 1)] = 1
                img_centers_all_all[cy - 3:cy + 3, cx - 3:cx + 3, :] = (255, 255, 255)
            e_dot_all = e_dot.copy()
            gt_centers = np.where(gt_dots_all > 0)
            for idx in range(len(gt_centers[0])):
                cx = gt_centers[1][idx]
                cy = gt_centers[0][idx]
                img_centers_all_all_gt[cy - 3:cy + 3, cx - 3:cx + 3, :] = (255, 255, 255)

            e_dot.astype(np.uint8).dump(
                os.path.join(img_dir, img_name.replace('.png', '_centers' + '_all' + '.npy')))
            if (visualize):
                '''
                    _centers_allcells.png
                        作用：显示所有细胞检测的中心点位置，包括检测到的和未检测到的
                        判断优劣：可以通过视觉检查检测到的中心点与实际中心点的一致性来评估细胞检测的准确性
                    _centers_det_overlay.png
                        作用：将检测到的中心点叠加在原始图像上，用于直观展示检测结果
                        判断优劣：可以通过观察检测到的中心点在图像上的分布和覆盖情况来评估细胞检测的全局效果
                    _allcells_hard.png
                        作用：显示应用了阈值后的二值细胞检测结果
                        判断优劣：可以通过比较二值检测结果与原始图像中实际细胞位置的一致性来评估二值化后的检测效果
                '''
                io.imsave(os.path.join(img_dir, img_name.replace('.png','_centers'+'_allcells' +'.png')), (e_dot_vis*255).astype(np.uint8))
                io.imsave(os.path.join(img_dir, img_name.replace('.png', '_centers' + '_det' + '_overlay.png')),
                          (img_centers_all_all).astype(np.uint8))
                io.imsave(os.path.join(img_dir, img_name.replace('.png','_allcells' +'_hard.png')), (e_hard2*255).astype(np.uint8))

            # 结束：评估所有的检测结果

            # 开始：评估分类
            et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
            e_hard2_all = e_hard2.copy()

            for s in range(n_classes):
                g_count = gt_dots[0, s, :, :].sum()

                e_hard2 = (et_class_argmax == s)

                # 用当前类别的预测结果过滤预测的检测点图
                e_dot = e_hard2 * e_dot_all
                e_count = e_dot.sum()

                g_dot = gt_dots[0, s, :, :].squeeze()
                e_dot_vis = np.zeros(g_dot.shape)
                e_dots_tuple = np.where(e_dot > 0)
                for idx in range(len(e_dots_tuple[0])):
                    cy = e_dots_tuple[0][idx]
                    cx = e_dots_tuple[1][idx]
                    img_centers_all[cy - 3:cy + 3, cx - 3:cx + 3, :] = color_set[s]

                gt_centers = np.where(g_dot > 0)
                for idx in range(len(gt_centers[0])):
                    cx = gt_centers[1][idx]
                    cy = gt_centers[0][idx]
                    img_centers_all_gt[cy - 3:cy + 3, cx - 3:cx + 3, :] = color_set[s]

                e_dot.astype(np.uint8).dump(
                    os.path.join(img_dir, img_name.replace('.png', '_centers' + '_s' + str(s) + '.npy')))
                '''
                    _likelihood_sX.png（X为类别编号）
                        作用：显示每个类别的检测概率图
                        判断优劣：对于每个类别，可以通过观察概率图来判断模型对该类别的识别效果，概率越高越好
                                （概率越高的区域一般会显示为越亮的颜色，通常是白色。
                                这是因为图像的亮度与概率值正相关，亮度越高表示模型更加确信对于位置属于该类别。）
                '''
                if(visualize):
                    io.imsave(os.path.join(img_dir, img_name.replace('.png','_likelihood_s'+ str(s)+'.png')), (et_class_sig.squeeze()[s]*255).astype(np.uint8));
            # 结束：评估所有的分类结果

            et_class_sig.squeeze().astype(np.float16).dump(
                os.path.join(img_dir, img_name.replace('.png', '_likelihood_class' + '.npy')))
            et_all_sig.squeeze().astype(np.float16).dump(
                os.path.join(img_dir, img_name.replace('.png', '_likelihood_all' + '.npy')))
            gt_dots.squeeze().astype(np.uint8).dump(
                os.path.join(img_dir, img_name.replace('.png', '_gt_dots_class' + '.npy')))
            gt_dots_all.squeeze().astype(np.uint8).dump(
                os.path.join(img_dir, img_name.replace('.png', '_gt_dots_all' + '.npy')))
            '''
                _centers_class_overlay.png:
                    作用：显示了模型对每个类别的检测中心的位置，并使用不同颜色表示不同的类别。
                         这些中心是通过对模型输出进行二进制阈值处理和连通组件分析获得的。
                    判断优劣：通过比较这些中心的位置与真实的标记（通过_gt_centers_class_overlay.png 查看）来评估模型的性能。
                             准确的中心位置意味着模型在检测中心方面表现良好
                _gt_centers_class_overlay.png:（真实值）
                    作用：显示了真实标记中每个类别的细胞中心的位置，使用不同颜色表示不同的类别。
                _likelihood_all.png：
                    作用：显示了模型对整个图像每个像素点属于所有类别的概率分布。通常，概率越高的区域呈现为亮的颜色。
                    判断优劣：这张图像可以用于全局评估模型的性能。你可以通过观察概率分布来确定模型对整个图像的分类是否准确。
                             概率分布应该在细胞中心附近有较高的值，而在其他地方应该有较低的值。
            '''
            if (visualize):
                io.imsave(os.path.join(img_dir, img_name.replace('.png', '_centers' + '_class_overlay' + '.png')),
                          (img_centers_all).astype(np.uint8))
                io.imsave(os.path.join(img_dir, img_name.replace('.png', '_gt_centers' + '_class_overlay' + '.png')),
                          (img_centers_all_gt).astype(np.uint8))
                io.imsave(os.path.join(img_dir, img_name), (img).astype(np.uint8))
                io.imsave(os.path.join(img_dir, img_name.replace('.png','_likelihood_all'+'.png')), (et_all_sig.squeeze()*255).astype(np.uint8))


            del img, gt_dots
