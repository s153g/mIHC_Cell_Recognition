import numpy as np
import time
import torch
import torch.nn as nn
import os

from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm  # 用于在循环中显示进度条
import sys;
import math
import skimage.io as io
import cv2
from skimage import filters
from skimage.measure import label, moments
import glob
from torch.utils.tensorboard import SummaryWriter
from skimage.feature import peak_local_max
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

# 导入自定义模块
from model_arch import UnetVggMultihead  # 导入神经网络模型

# ------------
# from my_dataloader_w_kfunc import CellsDataset  # 导入数据加载器
from my_dataloader import CellsDataset as CellsDataset_simple  # 导入简化版的数据加载器
from cluster_helper import *  # 导入辅助函数

# 定义各种文件和参数的路径
checkpoints_root_dir = '../MCSpatNet_checkpoints'  # 存储所有训练输出的根目录
checkpoints_folder_name = 'float_gaussian_test_chu_max'  # 在<checkpoints_root_dir>下创建用于保存当前训练实例输出的文件夹名称
attention_img_folder_name = 'attention_img'
model_param_path = None  # 用于继续训练的以前检查点的路径
# ---------------
# clustering_pseudo_gt_root = '../MCSpatNet_epoch_subclasses'  # 伪标签的根目录
train_data_root = 'D:\\GraduateWork\\code\\1\\Train12.8'  # 训练数据的根目录
test_data_root = 'D:\\GraduateWork\\code\\1\\Train12.8'  # 测试数据的根目录
train_split_filepath = 'D:\\GraduateWork\\code\\1\\Train12.8\\train_split.txt'  # 训练数据分割文件的路径
test_split_filepath = 'D:\\GraduateWork\\code\\1\\Train12.8\\val_split.txt'  # 测试数据分割文件的路径
epochs = 50  # 训练周期数

# 细胞类别和特征的编码
cell_code = {1:'CD8+T-bet+', 2:'CD8+T-bet-', 3:'CK+PD-L1+',4:'CK+PD-L1-'}

'''
    如若要去除细胞子类别分类和细胞交叉k函数模块的话，首先需要修改feature_code，其次需要修改model的n_heads和head_classes参数
'''
# 这个参数貌似只有聚类会用到
# --------------
# feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}


if __name__ == "__main__":

    # checkpoints_save_path: 保存检查点的路径
    checkpoints_save_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    # ---------------
    # cluster_tmp_out = os.path.join(clustering_pseudo_gt_root, checkpoints_folder_name)
    # attention_img_save_path: 注意力矩阵图的保存路径
    attention_img_save_path = os.path.join(checkpoints_save_path, attention_img_folder_name)

    # 如果不存在目录，则创建目录
    if not os.path.exists(checkpoints_root_dir):
        os.mkdir(checkpoints_root_dir)

    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)

    if not os.path.exists(attention_img_save_path):
        os.mkdir(attention_img_save_path)

    # --------------
    # if not os.path.exists(clustering_pseudo_gt_root):
    #     os.mkdir(clustering_pseudo_gt_root)
    #
    # if not os.path.exists(cluster_tmp_out):
    #     os.mkdir(cluster_tmp_out)

    writer = SummaryWriter(log_dir="./log/log_" + checkpoints_folder_name)

    # 日志文件路径
    i = 1
    while True:
        log_file_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name, f'train_log_{i}.txt')
        if(not os.path.exists(log_file_path)):
            break
        i += 1

    # 设置初始训练参数
    start_epoch             = 0  # 用于从以前的检查点中加载模型并继续训练的起始周期
    epoch_start_eval_prec   = 1  # 在epoch_start_eval_prec周期之后开始在验证集上评估F-score
    restart_epochs_freq     = 50 # 重置优化器的频率
    next_restart_epoch      = restart_epochs_freq + start_epoch
    gpu_or_cpu              ='cuda' # 使用cuda或cpu
    device=torch.device(gpu_or_cpu)
    seed                    = time.time()  # 随机数生成种子
    # print_frequency         = 1  # print frequency per epoch

    # 初始化日志文件
    log_file = open(log_file_path, 'a+')

    # 配置训练数据集
    train_image_root = os.path.join(train_data_root, 'images')  # 训练图像的根目录
    train_dmap_root = os.path.join(train_data_root, 'gt_custom')  # 训练目标密度图的根目录
    train_dots_root = os.path.join(train_data_root, 'gt_custom')  # 训练目标点图的根目录
    train_dmap_all_root = os.path.join(train_data_root, 'gt_custom')  # 训练目标密度图的根目录（用于细胞检测）

    # -------------
    # train_dmap_subclasses_root = cluster_tmp_out  # 子类别的伪标签密度图根目录
    # train_dots_subclasses_root = train_dmap_subclasses_root  # 子类别的伪标签点图根目录
    # train_kmap_root = os.path.join(train_data_root, 'k_func_maps')  # K函数地图的根目录

    # 配置验证数据集
    test_image_root = os.path.join(test_data_root, 'images')  # 验证图像的根目录
    test_dmap_root = os.path.join(test_data_root, 'gt_custom')  # 验证目标密度图的根目录
    test_dots_root = os.path.join(test_data_root, 'gt_custom')  # 验证目标点图的根目录
    test_dmap_all_root = os.path.join(test_data_root, 'gt_custom')  # 验证目标密度图的根目录（用于细胞检测）

    # -------------
    # test_dmap_subclasses_root = cluster_tmp_out  # 子类别的伪标签密度图根目录
    # test_dots_subclasses_root = test_dmap_subclasses_root  # 子类别的伪标签点图根目录
    # test_kmap_root = os.path.join(test_data_root, 'k_func_maps')  # 验证K函数地图的根目录

    # 定义一些超参数
    dropout_prob = 0.2  # 丢弃概率
    initial_pad = 126  # 添加填充以使最终输出与输入大小相同，因为不使用相同填充的卷积
    interpolate = 'False'
    conv_init = 'he'

    n_channels = 3
    n_classes = 4  # 细胞类别的数量
    n_classes_out = n_classes + 1  # 输出类别的数量 = 细胞类别数量 + 1（用于细胞检测通道）
    # s0-s3：四类细胞  s4：细胞检测通道

    # 9.8修改，npy文件的shape输出是(512,512,5)，其中4个不同类别的通道，加一个用于全部检测信息的通道
    class_indx = '1,2,3,4'  # 地面真实数据中类别通道的索引

    # ------------
    # n_clusters = 5  # 每个类别的群集数量
    # n_classes2 = n_clusters * (n_classes)  # 用于细胞群集分类的输出类别数量

    lr = 0.0005  # 学习率
    batch_size = 2
    prints_per_epoch = 1  # 每个epoch的打印频率

    # -----------------
    # # 初始化K函数的半径范围
    # r_step = 15
    # r_range = range(0, 100, r_step)
    # r_arr = np.array([*r_range])
    # r_classes = len(r_range)  # 单个类别的K函数的输出通道
    # r_classes_all = r_classes * (n_classes)  # 所有类别的K函数的输出通道数量

    # ----------
    # k_norm_factor = 100  # 归一化K函数的最大K值（即半径r处的附近细胞数量）到[0,1]

    lamda_dice = 1  # 主要输出通道（细胞检测 + 细胞分类）的Dice损失权重
    lamda_attention_loss = 1

    # -----------
    # lamda_subclasses = 1  # 次要输出通道(细胞群集分类)的Dice损失权重
    # lamda_k = 1  # K函数回归的L1损失权重

    # 设置随机种子
    torch.cuda.manual_seed(seed)

    # ----------------
    # 创建一个模型，参数配置保存在kwargs中
    # model = UnetVggMultihead(
    #     kwargs={'dropout_prob': dropout_prob,
    #             'initial_pad': initial_pad,
    #             'interpolate': interpolate,
    #             'conv_init': conv_init,
    #             'n_classes': n_classes,
    #             'n_channels': n_channels,
    #             'n_heads': 4,
    #             'head_classes': [1, n_classes, n_classes2, r_classes_all]
    #             }
    # )
    # 将输出头设置为2，去除聚类和交叉K函数
    model = UnetVggMultihead(
        kwargs={'dropout_prob': dropout_prob,
                'initial_pad': initial_pad,
                'interpolate': interpolate,
                'conv_init': conv_init,
                'n_classes': n_classes,
                'n_channels': n_channels,
                'n_heads': 2,
                'head_classes': [1, n_classes]
                }
    )
    # 如果提供了预训练模型的路径，则加载该模型
    if not (model_param_path is None):
        model.load_state_dict(torch.load(model_param_path), strict=False)
        log_file.write('model loaded \n')        
        log_file.flush()
    # 将模型移到GPU上
    model.to(device)

    # 初始化Sigmoid层用于细胞检测
    criterion_sig = nn.Sigmoid()
    # 初始化Softmax层用于细胞分类
    criterion_softmax = nn.Softmax(dim=1)

    # -------------
    # 初始化K函数的L1损失
    # criterion_l1_sum = nn.L1Loss(reduction='sum')

    # 初始化优化器
    optimizer = torch.optim.Adam(
        list(model.final_layers_lst.parameters())
        + list(model.decoder.parameters())
        + list(model.bottleneck.parameters())
        + list(model.encoder.parameters()),
        lr)

    # 初始化训练数据集加载器
    # train_dataset = CellsDataset(train_image_root,
    #                              train_dmap_root,
    #                              train_dots_root,
    #                              class_indx,
    #                              train_dmap_subclasses_root,
    #                              train_dots_subclasses_root,
    #                              train_kmap_root,
    #                              split_filepath=train_split_filepath,
    #                              phase='train',
    #                              fixed_size=448,
    #                              max_scale=16
    #                              )
    # 原始代码中，引用的是my_dataloader_w_kfunc.py的CellsDataset，现在更改为my_dataloader.py中的CellsDataset，即CellsDataset_simple
    train_dataset = CellsDataset_simple(train_image_root,
                                        train_dmap_root,
                                        train_dots_root,
                                        train_dmap_all_root,
                                        class_indx,
                                        split_filepath=train_split_filepath,
                                        phase='train',
                                        fixed_size=448,
                                        max_scale=16
                                        )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True  # 保证每次取得数据都和之前运行的一样
                                               )

    # 初始化验证数据集加载器
    # test_dataset = CellsDataset(test_image_root,
    #                             test_dmap_root,
    #                             test_dots_root,
    #                             class_indx,
    #                             test_dmap_subclasses_root,
    #                             test_dots_subclasses_root,
    #                             test_kmap_root,
    #                             split_filepath=test_split_filepath,
    #                             phase='test',
    #                             fixed_size=-1,
    #                             max_scale=16
    #                             )
    # 原始代码中，引用的是my_dataloader_w_kfunc.py的CellsDataset，现在更改为my_dataloader.py中的CellsDataset，即CellsDataset_simple
    test_dataset = CellsDataset_simple(test_image_root,
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

    # -----------------------
    # 初始化用于群集阶段的训练数据集加载器
    # simple_train_dataset = CellsDataset_simple(train_image_root,
    #                                            train_dmap_root,
    #                                            train_dots_root,
    #                                            class_indx,
    #                                            phase='test',
    #                                            fixed_size=-1,
    #                                            max_scale=16,
    #                                            return_padding=True
    #                                            )
    # simple_train_loader = torch.utils.data.DataLoader(simple_train_dataset,
    #                                                   batch_size=batch_size,
    #                                                   shuffle=False
    #                                                   )

    # 使用prints_per_epoch获取迭代次数以生成示例输出
    # print_frequency = len(train_loader)//prints_per_epoch
    print_frequency_test = len(test_loader) // prints_per_epoch  # print_frequency:41
    # print("test_loader len:", len(test_loader))

    # 初始话一些变量来跟踪最佳模型
    best_epoch_filepath = None
    best_epoch = None
    best_f1_mean = 0
    best_prec_recall_diff = math.inf

    # ---------------
    # centroids = None
    for epoch in range(start_epoch,epochs):
        # 如果epoch已经存在则跳过
        epoch_files = glob.glob(os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+"_*.pth"))
        if len(epoch_files) > 0:
            continue

        # -----------------------
        # 开始每个epoch的特征群集化
        # print('epoch', epoch, 'start clustering')
        # centroids = perform_clustering(model, simple_train_loader, n_clusters, n_classes, [feature_code['k-cell'], feature_code['subclass']], train_dmap_subclasses_root, centroids)
        # print('epoch', epoch, 'end clustering')
                
        # 训练阶段
        model.train()
        log_file.write('epoch= ' + str(epoch) + '\n')
        log_file.flush()

        # 初始化变量以累积整个epoch的损失
        epoch_loss = 0
        train_count = 0
        # train_loss_k = 0
        # train_loss_dice = 0
        # train_count_k = 0

        # ------------------
        for i,(img, gt_dmap, gt_dots, gt_dmap_all, img_name) in enumerate(tqdm(train_loader)):
            ''' 
                img: 输入图像
                gt_dmap: 细胞类别的地图的真实值，带有膨胀点
                gt_dots: 细胞类别的地图的真实值的二进制点地图
                gt_dmap_all: 细胞地图的真实值，带有膨胀点（用以细胞检测）
                remove gt_dmap_subclasses: 带有膨胀点的细胞群集子类别的真实地图。这可以是二进制掩码或密度地图（在这种情况下将被转换为二进制掩码）
                remove gt_dots_subclasses: 细胞群集子类别的真实二进制点地图
                remove gt_kmap: 地面真值 K 函数地图。每个细胞中心都包含以该细胞为中心的交叉 K 函数。
                img_name: 图像文件名
            '''
            # -------------
            # gt_kmap /= k_norm_factor  # 标准化K函数的地图
            # print(f"gt_dmap shape:{gt_dmap.shape}")
            # print(f"gt_dots shape:{gt_dots.shape}")

            img_name = img_name[0]
            train_count += 1

            img = img.to(device)
            # 将地图的真实值转换为二进制掩码（如果它们是密度地图的话）
            # gt_dmap = gt_dmap > 0
            # 将地图的真实值转换为二进制掩码，用于Attention loss的比较
            gt_dmap_all_binary = gt_dmap_all > 0
            gt_dmap_all_binary = gt_dmap_all_binary.type(torch.FloatTensor)
            gt_dmap_all_binary = gt_dmap_all_binary.to(device)

            # gt_dmap_all = gt_dmap_all.detach().cpu().numpy()
            # mean_value = np.mean(gt_dmap_all)
            # std_dev = np.std(gt_dmap_all)
            # min_value = np.min(gt_dmap_all)
            # max_value = np.max(gt_dmap_all)
            #
            # print(f"Mean: {mean_value}, Std Dev: {std_dev}, Min: {min_value}, Max: {max_value}")

            # -----------
            # gt_dmap_subclasses = gt_dmap_subclasses > 0
            # 从类别地图中获取检测地图的真实值（现在直接获得检测地图的真实值，即gt_dmap_all）
            # gt_dmap_all = gt_dmap.max(1)[0]
            # print(f"gt_dmap_all shape:{gt_dmap_all.shape}")

            # 设置数据类型并移到GPU
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            # print(f"gt_dmap shape:{gt_dmap.shape}")
            # print(f"gt_dmap_all shape:{gt_dmap_all.shape}")

            # ---------------
            # gt_dmap_subclasses = gt_dmap_subclasses.type(torch.FloatTensor)
            # gt_kmap = gt_kmap.type(torch.FloatTensor)
            gt_dmap = gt_dmap.to(device)
            gt_dmap_all = gt_dmap_all.to(device)

            # -----------
            # gt_dmap_subclasses=gt_dmap_subclasses.to(device)
            # gt_kmap=gt_kmap.to(device)

            # 前向传播
            et_dmap_lst, attention_coefficients = model(img)
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]  # 细胞检测预测
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]  # 细胞分类预测
            # 最后一层的注意力系数矩阵，并进行裁剪，上下左右各减少4个像素，以保证和img大小一致(经过Sigmoid的)
            attention_coefficient_last = attention_coefficients[3][:, :, 4:-4, 4:-4]

            # ----------------
            # et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2]  # 细胞群集子类别预测
            # et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # 交叉K函数估计

            # ----------------
            # 仅在检测掩模区域上应用K函数损失
            # k_loss_mask = gt_dmap_all.clone()
            # loss_l1_k = criterion_l1_sum(et_kmap*(k_loss_mask), gt_kmap*(k_loss_mask)) / (k_loss_mask.sum()*r_classes_all)

            # 对检测和分类预测应用Sigmoid和Softmax激活函数
            et_all_sig = criterion_sig(et_dmap_all)
            et_class_sig = criterion_softmax(et_dmap_class)
            # ------------------
            # et_subclasses_sig = criterion_softmax(et_dmap_subclasses)

            # 计算检测和分类预测的Dice损失
            intersection = (et_class_sig * gt_dmap).sum()
            union = (et_class_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice_class = 1 - ((2 * intersection + 1) / (union + 1))

            intersection = (et_all_sig * gt_dmap_all.unsqueeze(0)).sum()
            union = (et_all_sig**2).sum() + (gt_dmap_all.unsqueeze(0)**2).sum()
            loss_dice_all = 1 - ((2 * intersection + 1) / (union + 1))

            '''
                以SSIM的方式来计算Attention Loss效果并不是很好
            '''
            # # 将 PyTorch 张量转换为 NumPy 数组，将attention_coefficient_last的第二个维度取消掉
            # attention_coefficient_last_np = attention_coefficient_last.squeeze(1).detach().cpu().numpy()
            # gt_dmap_all_binary_np = gt_dmap_all_binary.detach().cpu().numpy()
            #
            # # 初始化一个数组来存储每个图像的SSIM值
            # ssim_values = np.zeros(attention_coefficient_last_np.shape[0])
            # # print("attention_coefficient_last_np:", str(i) + '  ' + str(attention_coefficient_last_np.shape))
            # # print("gt_dmap_all_np:", str(i) + '  ' + str(gt_dmap_all_np.shape))
            # # 循环计算每个图像的SSIM
            # for num in range(attention_coefficient_last_np.shape[0]):
            #     ssim_values[num] = ssim(attention_coefficient_last_np[num], gt_dmap_all_binary_np[num])
            #
            # # 计算结构相似性
            # attention_loss = 1 - np.mean(ssim_values)
            # # print("attention_loss:", attention_loss)

            '''
                以MSE（平均平方误差）来计算Attention Loss进行尝试
            '''
            attention_loss = F.mse_loss(attention_coefficient_last.squeeze(1), gt_dmap_all_binary)

            # ----------------
            # intersection = (et_subclasses_sig * gt_dmap_subclasses ).sum()
            # union = (et_subclasses_sig**2).sum() + (gt_dmap_subclasses**2).sum()
            # loss_dice_subclass =  1 - ((2 * intersection + 1) / (union + 1))

            # --------------
            # loss_dice = loss_dice_class + loss_dice_all + lamda_subclasses * loss_dice_subclass
            loss_dice = loss_dice_class + loss_dice_all + lamda_attention_loss * attention_loss
            # train_loss_dice += loss_dice.item()

            # 添加Dice损失和K函数L1损失。如果K函数为NAN，尤其是在训练开始时，不要添加到损失中。
            loss = (lamda_dice * loss_dice)
            # --------------
            # if(not math.isnan(loss_l1_k.item())):
            #     loss += loss_l1_k * lamda_k
            #     # train_count_k += 1
            #     # train_loss_k += loss_l1_k.item()

            # 反向传播损失
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---------------------
            # log_file.write("epoch: "+str(epoch)
            #                + "  i: "+str(i)
            #                + "   loss_dice: "+str(loss_dice.item())
            #                + "   loss_l1_k:" +str(loss_l1_k.item())
            #                + '\n'
            #                )

            log_file.write("epoch: " + str(epoch)
                           + "  i: " + str(i)
                           + "   loss_dice: " + str(loss_dice.item())
                           + '\n'
                           )
            log_file.flush()


        log_file.write("epoch: " + str(epoch)
                       + " train loss: " + str(epoch_loss/train_count)
                       + '\n'
                       )
        log_file.flush()
        epoch_loss = epoch_loss/train_count

        writer.add_scalar('train loss', epoch_loss/train_count, epoch)
        # break

        # 测试验证集上的模型性能，这里应该使用验证集，一旦测试集参与了训练，就不能用作测试了
        model.eval()
        err = np.array([0 for s in range(n_classes_out)])
        loss_val = 0
        loss_val_k_wo_nan = 0
        loss_val_k = 0
        loss_val_dice = 0
        loss_val_dice2 = 0
        tp_count_all = np.zeros(n_classes_out)
        fp_count_all = np.zeros(n_classes_out)
        fn_count_all = np.zeros(n_classes_out)
        test_count_k = 0
        """
            12.2  准备添加gt_dmap_all参数
        """
        for i, (img, gt_dmap, gt_dots, gt_dmap_all, img_name) in enumerate(tqdm(test_loader)):
            ''' 
                img: 输入图像
                gt_dmap: 细胞类别的地图的真实值，带有膨胀点
                gt_dots: 细胞类别的地图的真实值的二进制点地图
                gt_dmap_all: 全部细胞地图的真实值，带有膨胀点（用以细胞检测）
                remove gt_dmap_subclasses: 带有膨胀点的细胞群集子类别的真实地图。这可以是二进制掩码或密度地图（在这种情况下将被转换为二进制掩码）
                remove gt_dots_subclasses: 细胞群集子类别的真实二进制点地图
                remove gt_kmap: 地面真值 K 函数地图。每个细胞中心都包含以该细胞为中心的交叉 K 函数。
                img_name: 图像文件名
            '''
            # ------------------
            # gt_kmap /= k_norm_factor  # 标准化 K 函数的地图
            img_name = img_name[0]
            img = img.to(device)
            # 将地图的真实值转换为二进制掩码（如果它们是密度地图的话）
            # gt_dmap = gt_dmap > 0
            # 从类别地图中获取检测地图的真实值
            # gt_dmap_all = gt_dmap.max(1)[0]
            gt_dots_all = gt_dots.max(1)[0]
            # 设置数据类型并移到 GPU
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            # ------------------------
            # gt_kmap = gt_kmap.type(torch.FloatTensor)
            # gt_kmap=gt_kmap.to(device)
            # k_loss_mask = gt_dmap_all.clone().to(device)     # 仅在膨胀点掩码上应用 K 函数损失

            # 将地图的真实值转换为NumPy数组
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_all = gt_dots_all.detach().cpu().numpy()
            gt_dmap = gt_dmap.detach().cpu().numpy()
            gt_dmap_all = gt_dmap_all.detach().cpu().numpy()

            # 前向传播
            et_dmap_lst, attention_coefficients = model(img)
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]  # The cell detection prediction
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]  # The cell classification prediction
            # -------------------
            # et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2] # The cell clustering sub-class prediction
            # et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # The cross K-functions estimation

            writer.add_histogram('et_all_sig', criterion_sig(et_dmap_all), global_step=0)

            # 对检测和分类预测应用 Sigmoid 和 Softmax 激活函数
            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            # ----------------
            # 仅在检测掩码区域上应用 K 函数损失
            # loss_l1_k = criterion_l1_sum(et_kmap*(k_loss_mask), gt_kmap*(k_loss_mask)) / (k_loss_mask.sum()*r_classes_all)

            # 保存样本输出预测，只有整除时，才输出，在该测试集下，print_frequency_test是41，即只输出能被41整除的图像
            if(i % print_frequency_test == 0):
                # attention_coefficients 是一个列表，其中包含了多个注意力系数矩阵
                for layer, attention_matrix in enumerate(attention_coefficients):
                    if layer == 3:
                        # 如果是最后一层，需要进行裁剪以保证和原图像尺寸一致
                        attention_matrix = attention_matrix[:, :, 4:-4, 4:-4]

                    # 将 PyTorch Tensor 转换为 NumPy 数组
                    attention_array = attention_matrix.detach().cpu().numpy()

                    # 归一化到 0-255 的范围
                    normalized_attention = (attention_array[0] - attention_array[0].min()) / (
                            attention_array[0].max() - attention_array[0].min()) * 255

                    # 创建 PIL Image 对象，'L'表示单通道灰度图像
                    attention_image = Image.fromarray(np.squeeze(normalized_attention).astype(np.uint8), mode='L')

                    # 保存为 PNG 文件
                    image_path = os.path.join(attention_img_save_path, f'attention_epoch{epoch}_test_indx{i}_layer_{layer}.png')
                    attention_image.save(image_path)

                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_img'+'.png'), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8))
                for s in range(n_classes):
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_test'+ '_indx'+str(i)+'_likelihood'+'_s'+str(s)+'.png'), (et_class_sig[:,s,:,:]*255).squeeze().astype(np.uint8))
                    io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_gt'+'_s'+str(s)+'.png'), (gt_dmap[:,s,:,:]*255).squeeze().astype(np.uint8))
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_test'+ '_indx'+str(i)+'_likelihood'+'_all'+'.png'), (et_all_sig*255).squeeze().astype(np.uint8))
                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_gt'+'_all'+'.png'), (gt_dmap_all*255).squeeze().astype(np.uint8))

            # -----------------
            # 累积 K 函数测试损失
            # loss_val_k += loss_l1_k.item()
            # if(not math.isnan(loss_l1_k.item())):
            #     loss_val_k_wo_nan += loss_l1_k.item()
            #     test_count_k += 1

            intersection = (et_class_sig * gt_dmap).sum()
            union = (et_class_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice_class = 1 - ((2 * intersection + 1) / (union + 1))

            intersection = (et_all_sig * gt_dmap_all).sum()
            union = (et_all_sig**2).sum() + (gt_dmap_all**2).sum()
            loss_dice_all = 1 - ((2 * intersection + 1) / (union + 1))

            loss_dice = (loss_dice_class + loss_dice_all).item()
            loss_val_dice += loss_dice

            # --------------
            # print('epoch', epoch, 'test', i, 'loss_l1_k', str(loss_l1_k.item()), 'loss_dice', str(loss_dice))
            print('epoch', epoch, 'test', i, 'loss_dice', str(loss_dice))

            writer.add_scalar('loss_diss', loss_dice, epoch)

            # 如果 epoch >= epoch_start_eval_prec，则计算 F-分数
            if(epoch >= epoch_start_eval_prec):
                # 对检测输出应用 0.5 阈值并转换为二进制掩码
                e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), 0.5, 0.5)
                e_hard2 = (e_hard > 0).astype(np.uint8)
                e_hard2_all = e_hard2.copy()

                # 对e_hard2进行距离变换，并获取其局部极大值的坐标，从而将粘连细胞分离
                # 进行距离变换，并可视化结果
                e_hard2 = cv2.distanceTransform(e_hard2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                # 将距离变换结果归一化到[0,1]
                cv2.normalize(e_hard2, e_hard2, 0, 1, cv2.NORM_MINMAX)

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
                e_hard2 = markers[20:e_hard2.shape[0] + 20, 20:e_hard2.shape[1] + 20]
                e_hard2 = (e_hard2 > 0).astype(np.uint8)

                # 通过查找二进制掩码中轮廓的中心来获取预测的细胞中心
                e_dot = np.zeros((img.shape[-2], img.shape[-1]))
                contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for idx in range(len(contours)):
                    contour_i = contours[idx]
                    M = cv2.moments(contour_i)
                    if(M['m00'] == 0):
                        continue;
                    cx = round(M['m10'] / M['m00'])
                    cy = round(M['m01'] / M['m00'])
                    e_dot[cy, cx] = 1
                e_dot_all = e_dot.copy()

                tp_count = 0  # 初始化真正例数(模型正确的正预测)
                fp_count = 0  # 初始化假正例数(模型错误的正预测)
                fn_count = 0  # 初始化假负例数(模型错误的负预测)
                # 将 g_dot_vis 初始化为包含所有细胞检测地图的地面真正例点
                g_dot_vis = gt_dots_all.copy().squeeze()
                # 获取预测的检测二进制地图中的连通组件
                e_hard2_comp = label(e_hard2)
                e_hard2_comp_all = e_hard2_comp.copy()
                # 对于每个连通组件，如果它与地面真正例点相交，则它是真正例；否则，它是假正例。
                # 如果它是真正例，则从 g_dot_vis 中删除它。
                # 注意：如果有多个地面真正例点相交，那么只有一个是真正例。
                for l in range(1, e_hard2_comp.max()+1):
                    e_hard2_comp_l = (e_hard2_comp == l)
                    M = moments(e_hard2_comp_l)
                    (y,x) = int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0])
                    if ((e_hard2_comp_l * g_dot_vis).sum()>0): # 真正例
                        tp_count += 1
                        (yg, xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                        yg = yg[0]
                        xg = xg[0]
                        g_dot_vis[yg, xg] = 0
                    else: #((e_hard2_comp_l * g_dot_vis).sum()==0): # 假正例
                        fp_count += 1
                # g_dot_vis 中剩余的细胞是假负例。
                fn_points = np.where(g_dot_vis > 0)
                fn_count = len(fn_points[0])

                # 从当前图像预测的真正例、假正例和假负例数量更新 TP、FP 和 FN 计数
                tp_count_all[-1] = tp_count_all[-1] + tp_count
                fp_count_all[-1] = fp_count_all[-1] + fp_count
                fn_count_all[-1] = fn_count_all[-1] + fn_count

                # argmax()返回输入张量中所有元素中最大值对应的索引（按行搜索）
                # 获取预测的细胞类别
                et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
                e_hard2_all = e_hard2.copy()
                # 对于每个类别，获取真正例、假正例和假负例计数，类似于先前的检测代码
                for s in range(n_classes):
                    g_count = gt_dots[0,s,:,:].sum()

                    e_hard2 = (et_class_argmax == s)  
                
                    e_dot = e_hard2 * e_dot_all  

                    g_dot = gt_dots[0, s,:,:].squeeze()

                    tp_count = 0
                    fp_count = 0
                    fn_count = 0
                    g_dot_vis = g_dot.copy()
                    e_dots_tuple = np.where(e_dot > 0)
                    for idx in range(len(e_dots_tuple[0])):
                        cy=e_dots_tuple[0][idx]
                        cx=e_dots_tuple[1][idx]
                        l = e_hard2_comp_all[cy, cx]
                        e_hard2_comp_l = (e_hard2_comp == l)
                        if ((e_hard2_comp_l * g_dot_vis).sum()>0): # 真正例
                            tp_count += 1
                            (yg,xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                            yg = yg[0]
                            xg = xg[0]
                            g_dot_vis[yg,xg] = 0 
                        else: #((e_hard2_comp_l * g_dot_vis).sum()==0): # 假正例
                            fp_count += 1
                    fn_points = np.where(g_dot_vis > 0)
                    fn_count = len(fn_points[0])


                    tp_count_all[s] = tp_count_all[s] + tp_count
                    fp_count_all[s] = fp_count_all[s] + fp_count
                    fn_count_all[s] = fn_count_all[s] + fn_count


            del img, gt_dmap, gt_dmap_all, et_dmap_all, et_dmap_class, gt_dots


        saved = False

        # 初始化三个空数组来存储每个类别的精确度、召回率和F1值
        precision_all = np.zeros((n_classes_out))
        recall_all = np.zeros((n_classes_out))
        f1_all = np.zeros((n_classes_out))

        # 如果当前 epoch 大于等于评估开始的 epoch
        if(epoch >= epoch_start_eval_prec):
            # 计算所有类别中真正例和假负例的总和
            count_all = tp_count_all.sum() + fn_count_all.sum()
            # 针对每个类别执行以下操作
            for s in range(n_classes_out):
                # 如果真正例和假正例的总和为零，将精确度设置为 1（避免除以零）
                if(tp_count_all[s] + fp_count_all[s] == 0):
                    precision_all[s] = 1
                else:
                    precision_all[s] = tp_count_all[s]/(tp_count_all[s] + fp_count_all[s])

                # 如果真正例和假负例的总和为零，将召回率设置为 1（避免除以零）
                if(tp_count_all[s] + fn_count_all[s] == 0):
                    recall_all[s] = 1
                else:
                    recall_all[s] = tp_count_all[s]/(tp_count_all[s] + fn_count_all[s])

                # 如果精确度和召回率之和为零，将 F1 值设置为零（避免除以零）
                if(precision_all[s]+recall_all[s] == 0):
                    f1_all[s] = 0
                else:
                    # 计算 F1 值，这是精确度和召回率的调和平均值
                    f1_all[s] = 2*(precision_all[s] *recall_all[s])/(precision_all[s]+recall_all[s])
                # 打印当前类别的性能指标
                print_msg = f'epoch {epoch} s {s} precision_all {precision_all[s]} recall_all {recall_all[s]} f1_all {f1_all[s]}'
                print(print_msg)

                # 将性能指标写入日志文件
                log_file.write(print_msg+'\n')
                log_file.flush()
            # 计算并打印所有类别的平均性能指标
            print_msg = f'epoch {epoch} all precision_all {precision_all.mean()} recall_all {recall_all.mean()} f1_all {f1_all.mean()}'
            print(print_msg)
            log_file.write(print_msg+'\n')
            log_file.flush()

            # 计算并打印除了最后一个类别之外的平均性能指标
            print_msg = f'epoch {epoch} classes precision_all {precision_all[:-1].mean()} recall_all {recall_all[:-1].mean()} f1_all {f1_all[:-1].mean()}'
            print(print_msg)
            log_file.write(print_msg+'\n')
            log_file.flush()

            writer.add_scalar('classes precision_all', precision_all[:-1].mean(), epoch)
            writer.add_scalar('classes recall_all', recall_all[:-1].mean(), epoch)
            writer.add_scalar('classes f1_all', f1_all[:-1].mean(), epoch)

        writer.close()
        # 检查当前 epoch 是否是最佳 epoch，基于验证集上的 F1 分数
        model_save_postfix = ''
        is_best_epoch = False
        # if (f1_all.mean() > best_f1_mean):
        if (f1_all.mean() - best_f1_mean >= 0.005):
            model_save_postfix += '_f1'
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
            is_best_epoch = True
        # 或者，如果当前 F1 平均值和历史最佳值之间的差异小于 0.005（稍微低一点的 F 分数），但精确度和召回率之间的差异更小
        elif ((abs(f1_all.mean() - best_f1_mean) < 0.005) # a slightly lower f score but smaller gap between precision and recall
                and abs(recall_all.mean()-precision_all.mean()) < best_prec_recall_diff):
            model_save_postfix += '_pr-diff'
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
            is_best_epoch = True
        # if (recall_all.mean() > best_recall_mean):
        #     model_save_postfix += '_rec'
        #     best_recall_mean = recall_all.mean()
        #     is_best_epoch = True

        # 保存模型检查点如果是最佳 epoch
        if((saved == False) and (model_save_postfix != '')):
            print('epoch', epoch, 'saving')

            # 构建新的检查点文件路径
            new_epoch_filepath = os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+model_save_postfix+".pth")
            # 保存模型权重
            torch.save(model.state_dict(), new_epoch_filepath ) # save only if get better error
            # 保存聚类的中心点数据
            # centroids.dump(os.path.join(checkpoints_save_path, 'epoch{}_centroids.npy'.format(epoch)))

            saved = True
            print_msg = f'epoch {epoch} saved.'
            print(print_msg)

            # 将保存信息写入日志文件
            log_file.write(print_msg+'\n')
            log_file.flush()
            # 如果是最佳 epoch，将当前文件路径设为最佳 epoch 文件路径
            if(is_best_epoch):
                best_epoch_filepath = new_epoch_filepath
                best_epoch = epoch

        # 重置 Adam 优化器以避免参数学习速率下降
        sys.stdout.flush()
        if((epoch >= next_restart_epoch) and not(best_epoch_filepath is None)):
            next_restart_epoch = epoch + restart_epochs_freq
            model.load_state_dict(torch.load(best_epoch_filepath), strict=False)
            model.to(device)
            # 重新初始化优化器
            optimizer = torch.optim.Adam(list(model.final_layers_lst.parameters())+list(model.decoder.parameters())+list(model.bottleneck.parameters())+list(model.encoder.parameters()),lr)

    log_file.close()
    
