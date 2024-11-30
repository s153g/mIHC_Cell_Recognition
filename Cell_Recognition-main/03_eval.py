import os
import numpy as np
import glob
import sys
import scipy.spatial
from skimage import io;
from scipy.ndimage.filters import convolve   
from scipy import ndimage

"""
    距离阈值的含义是用来判断预测点与真实点之间的距离是否小于或等于该阈值。如果是，就认为这两个点匹配。
    
    在该文件中，有一个嵌套的循环，其中外部循环是对不同的距离阈值进行迭代的，而内部循环则是对每个真值点进行迭代的。
    对于每个真值点，算法会在预测点中找到距离真值点最近的点，然后判断这个距离是否小于当前的距离阈值。
    如果是的话，就将这个预测点认为是真值点的匹配，否则认为是漏检。
    
    输出有class0-class4，其中class0-class3对应四类细胞，class4对应细胞检测
"""

# 配置参数
# # 数据目录，包含了真值和预测结果
data_dir = '../MCSpatNet_eval/mcspatnet_consep_2_e18'

# 计算距离阈值的F分数，范围在(1, max_dist_thresh)之间
max_dist_thresh = 8  # 6
# mpp = 0.254 at 40x,  ppm at 20x = 1/(0.254*2),  mpp at 20x = 0.254*2 = 0.508, 6 px = 0.508*6 = 3.048 microns, , 30 px = 0.508*30=15.24 microns

# 定义颜色集合，用于可视化
# color_set = {'CD8+T-bet+': (0, 162, 232), 'CD8+T-bet-': (255, 0, 0), 'CK+PD-L1+': (0, 255, 0), 'CK+PD-L1-': (255, 255, 0)}
color_set = {'tp':(0,162,232),'fp':(0,255,0),'fn':(255,255,0)}

def calc(g_dot, e_dot, class_indx, img_indx, img_name):

    '''
        根据不同的距离阈值计算类别 class_indx 的TP、FP、FN数量
        对于阈值 t，TP 预测是在距离真值预测 t 像素内的预测，这些真值预测之前未被处理过。
    '''

    leafsize = 2048  # KD树的叶节点数
    k = 50  # 最近邻的点数
    e_coords = np.where(e_dot > 0)  # 获取预测点的坐标
    # 从预测的细胞中心构建KD树
    z = np.zeros((len(e_coords[0]), 2))
    z[:,0] = e_coords[0]
    z[:,1] = e_coords[1]
    if(len(e_coords[0]) > 0):
        tree = scipy.spatial.KDTree(z, leafsize=leafsize)  # 构建KD树
        print('tree.data.shape', tree.data.shape)

    for dist_thresh in range(1, max_dist_thresh+1):
        img_f = np.zeros((e_dot.shape[0],e_dot.shape[1],3))  # 初始化用于可视化结果的图像矩阵
        print('class_indx', class_indx, 'thresh', dist_thresh, 'len(e_coords[0])',len(e_coords[0]))
        if(len(e_coords[0]) == 0):  # 没有预测结果的情况
            for dist_thresh in range(1,max_dist_thresh+1):
                tp_img = 0
                fn_img = (g_dot > 0).sum()  # 计算真值中的正样本数量
                fp_img = 0
                fn[class_indx, dist_thresh] += fn_img  # 更新FN数量
        else:
            tp_img = 0
            fn_img = 0
            fp_img = 0

            e_dot_processing = np.copy(e_dot)  # 复制预测结果

            gt_points = np.where(g_dot > 0)  # 获得真值点的坐标
            ''' 
                遍历真值点并找到距离阈值内的最近预测
                    如果有匹配并且匹配点存在于 e_dot_processing 中，
                        则这是一个TP，从 e_dot_processing 中删除匹配点，以便每个预测只匹配一次。
                    否则
                        这是一个FN
                e_dot_processing 中剩余的预测被计为FP                
            '''
            for pi in range(len(gt_points[0])):
                p = [[gt_points[0][pi], gt_points[1][pi]]]
                distances, locations = tree.query(p, k=k,distance_upper_bound =dist_thresh)
                match = False
                for nn in range(min(k,len(locations[0]))):
                    if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) and (e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] > 0)):
                    #if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) ):
                        tp[class_indx, dist_thresh] += 1
                        tp_img +=1
                        e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] = 0
                        match = True
                        py = int(tree.data[locations[0][nn]][0])
                        px = int(tree.data[locations[0][nn]][1])
                        img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['tp']
                        break
                if(not match):
                    fn[class_indx, dist_thresh] += 1
                    fn_img +=1
                    py = gt_points[0][pi]
                    px = gt_points[1][pi]
                    img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fn']

            fp[class_indx, dist_thresh] += e_dot_processing.sum()
            fp_img +=e_dot_processing.sum()
            fp_points = np.where(e_dot_processing > 0)
            for pi in range(len(fp_points[0])):
                py = fp_points[0][pi]
                px = fp_points[1][pi]
                img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fp']
            io.imsave(os.path.join(out_dir, img_name+'_s'+str(class_indx)+'_f'+'_th'+str(dist_thresh)+'.png'), img_f.astype(np.uint8))
            print(img_name, 's',class_indx, 'thresh', dist_thresh, 'tp', tp_img, 'fp', fp_img, 'fn', fn_img)
            sys.stdout.flush();

        # 计算当前图像和阈值的精度、召回率和F分数
        if(tp_img + fp_img == 0):
            precision_img[class_indx, dist_thresh, img_indx] = 1
        else:
            precision_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fp_img)
        if(tp_img + fn_img == 0):
            recall_img[class_indx, dist_thresh, img_indx] = 1
        else:
            recall_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fn_img) # True pos rate
        if(precision_img[class_indx, dist_thresh, img_indx] + recall_img[class_indx, dist_thresh, img_indx] == 0):
            f1_img[class_indx, dist_thresh, img_indx] = 0
        else:
            f1_img[class_indx, dist_thresh, img_indx] = 2*(( precision_img[class_indx, dist_thresh, img_indx]*recall_img[class_indx, dist_thresh, img_indx] )/( precision_img[class_indx, dist_thresh, img_indx]+recall_img[class_indx, dist_thresh, img_indx] ))


def eval(data_dir, out_dir):
    '''
        假设细胞类别的真值点地图在与预测文件相同的目录中。
        真值点地图的命名方式为 <img name>_gt_dots_class.npy
        预测点地图的命名方式为 <img name>_centers_s<class indx>.npy（用于分类）和 <img name>_centers_allcells.npy（用于检测）
    '''

    img_indx=-1
    with open(os.path.join(out_dir, 'out_distance_scores.txt'), 'a+') as log_file:
        for gt_filepath in gt_files:
            img_indx += 1
            print('gt_filepath',gt_filepath)
            sys.stdout.flush()
            img_name = os.path.basename(gt_filepath)[:-len('_gt_dots_class.npy')]
            g_dot_arr=np.load(gt_filepath, allow_pickle=True)

            # 处理细胞分类
            for s in range(n_classes):
                e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_s'+str(s)+'.npy'))[0]
                print('e_soft_filepath', e_soft_filepath)
                class_indx = s
                g_dot = g_dot_arr[class_indx]
                #print('e_soft_filepath',e_soft_filepath)
                sys.stdout.flush()
                e_dot = np.load(e_soft_filepath, allow_pickle=True)
                e_dot_vis = ndimage.convolve(e_dot, np.ones((5,5)), mode='constant', cval=0.0)            
                io.imsave(os.path.join(data_dir,img_name + '_centers_s' + str(s) + '_et.png'),(e_dot_vis*255).astype(np.uint8))
                calc(g_dot, e_dot, class_indx, img_indx, img_name)

            # 处理细胞检测
            e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_all*.npy'))[0]
            class_indx += 1
            g_dot = g_dot_arr.max(axis=0)
            #print('e_soft_filepath',e_soft_filepath)
            sys.stdout.flush()
            e_dot = np.load(e_soft_filepath, allow_pickle=True)
            calc(g_dot, e_dot, class_indx, img_indx, img_name)



        # tp.astype(np.int).dump(os.path.join(out_dir, 'tp.npy'))
        # fp.astype(np.int).dump(os.path.join(out_dir, 'fp.npy'))
        # fn.astype(np.int).dump(os.path.join(out_dir, 'fn.npy'))

        # 计算每个类别（类别索引在范围（0，n_classes-1）内）以及检测任务（类别索引=n_classes）在范围（1，max_dist_thresh）内的每个距离阈值的精度、召回率和F分数
        for class_indx in range(n_classes_out):
            for dist_thresh in range(1,max_dist_thresh+1):
                if(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh] == 0):
                    precision[class_indx, dist_thresh] = 1
                else:
                    precision[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh])
                if(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh] == 0):
                    recall[class_indx, dist_thresh] = 1
                else:
                    recall[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh]) # True pos rate
                if(precision[class_indx, dist_thresh] + recall[class_indx, dist_thresh] == 0):
                    f1[class_indx, dist_thresh] = 0
                else:
                    f1[class_indx, dist_thresh] = 2*((precision[class_indx, dist_thresh]*recall[class_indx, dist_thresh])/(precision[class_indx, dist_thresh]+recall[class_indx, dist_thresh]))

                print('class', class_indx, 'thresh', dist_thresh, 'prec', precision[class_indx, dist_thresh], 'recall', recall[class_indx, dist_thresh], 'fscore',f1[class_indx, dist_thresh])
                log_file.write("class {} thresh {} prec {} recall {} fscore {}\n".format(class_indx, dist_thresh, precision[class_indx, dist_thresh], recall[class_indx, dist_thresh], f1[class_indx, dist_thresh]))


            log_file.flush()

if __name__ == "__main__":
    out_dir = data_dir  # 可以更改输出目录
    n_classes = 4  # 类别数量
    n_classes_out = n_classes + 1  # 输出包括细胞类别和细胞检测

    # 初始化统计变量
    tp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fn = np.zeros((n_classes_out, max_dist_thresh + 1))
    precision = np.zeros((n_classes_out, max_dist_thresh + 1))
    recall = np.zeros((n_classes_out, max_dist_thresh + 1))
    f1 = np.zeros((n_classes_out, max_dist_thresh + 1))

    gt_files = glob.glob(os.path.join(data_dir, '*_gt_dots_class'+'.npy'))
    #gt_files = glob.glob(os.path.join(data_dir, '*test_1_gt_dots_class'+'.npy'))
    print('len(gt_files)',len(gt_files))

    precision_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    recall_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    f1_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))

    eval(data_dir, out_dir)



