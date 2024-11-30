import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import collections
from distutils.util import strtobool;
import numpy as np  

from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;

"""
    float_mix_attention_gate_g_ConvTranspose2d实验的备份文件
    2024.3.1
        和float_mix_attention_gate_DataParallel设置基本一致，除了修改了Attention Gate模块中g的输入来源。
        在float_mix_attention_gate_DataParallel中，g的来源是下一层的解码器的输出。
        在该实验中，g的来源是下一层解码器的输出经过上采样操作的输出。
        
    2024.3.2
        实验发现，g的来源是下一层解码器的输出经过上采样操作的输出，效果更好，因此将在该实验的基础上完成后续实验。
        将瓶颈层的512通道调整成1024通道进行实验。（需要注释掉给瓶颈层初始化权重的操作）
        
    2024.3.3
        实验发现，取消了预训练权重加载后，模型可能需要更多的训练迭代才能收敛到最佳性能。这也解释了为什么取消预训练后，F1 分数下降了1.5% 并且在更多的 epochs 之后才达到最佳性能。
        所以，暂时取消1024通道的实验，并修改回512通道。
        
    2024.3.8
        forward函数返回了注意力系数矩阵，以列表形式，将四个注意力系数矩阵都保存起来进行返回
        
"""
class UnetVggMultihead(nn.Module):
    def __init__(self, load_weights=False, kwargs=None):
        super(UnetVggMultihead,self).__init__()

        # 预定义参数列表
        args = {'conv_init': 'he', 'block_size': 3, 'pool_size': 2
            , 'dropout_prob': 0, 'initial_pad': 0, 'n_classes': 1, 'n_channels': 3, 'n_heads': 2, 'head_classes': [1, 1]
        }

        if not(kwargs is None):
            args.update(kwargs)

        # 四种不同的权重初始化方法
        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # 读取额外的参数
        self.n_channels = int(args['n_channels'])  # 输入图像的通道数
        self.n_classes = int(args['n_classes'])    # 分类头
        self.conv_init = str(args['conv_init']).lower()  # 卷积层参数初始化方式
        self.n_heads = int(args['n_heads'])        # 并行处理的头数
        self.head_classes = np.array(args['head_classes']).astype(int)  # 每个头的分类数
    
        self.block_size = int(args['block_size'])  # 卷积核大小
        self.pool_size = int(args['pool_size'])    # 池化核大小
        self.dropout_prob = float(args['dropout_prob'])  # dropout概率
        self.initial_pad = int(args['initial_pad'])  # 初始填充

        #print('self.initial_pad',self.initial_pad)

        # Contracting Path (Encoder + Bottleneck)
        self.encoder = nn.Sequential()  # 编码器
        layer_index = 0
        layer = nn.Sequential()
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(self.n_channels, 64, kernel_size=self.block_size, padding=self.initial_pad))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(64, 64, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer)

        layer_index = 1
        layer = nn.Sequential()
        # MaxPool2d：下采样
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(64, 128, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(128, 128, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer)

        layer_index = 2
        layer = nn.Sequential()
        # MaxPool2d：下采样
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(128, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer)

        layer_index = 3
        layer = nn.Sequential()
        # MaxPool2d：下采样
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob))
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(256, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer)

        self.bottleneck = nn.Sequential()  # 瓶颈层
        # MaxPool2d：下采样
        self.bottleneck.add_module('bottleneck_maxpool', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size))
        self.bottleneck.add_module('bottleneck_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob))
        self.bottleneck.add_module('bottleneck_conv'+ '_0', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu'+'_0', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu'+'_1', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size))
        self.bottleneck.add_module('bottleneck_relu'+'_2', nn.ReLU(inplace=True))
     

        # Expanding Path (Decoder)
        self.decoder = nn.Sequential()  # 解码器
        layer_index = 3
        layer = nn.Sequential()
        # ConvTranspose2d：上采样。ConvTranspose2d 是 PyTorch 中的转置卷积（也称为反卷积）操作。
        # 这个操作的作用是将输入张量的空间维度扩大，通常用于上采样。
        # 具体来说，ConvTranspose2d 在输入上应用卷积核，但是与标准卷积相反，它会填充输出，从而使输出的尺寸大于输入的尺寸。
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 512, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_attention_'+str(layer_index), AttentionGate([512, 512], 512))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(1024, 512, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(512, 512, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_'+str(layer_index), layer)

        layer_index = 2
        layer = nn.Sequential()
        # ConvTranspose2d：上采样
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 256, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_attention_' + str(layer_index), AttentionGate([256, 256], 256))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(512, 256, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(256, 256, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_'+str(layer_index), layer)

        layer_index = 1
        layer = nn.Sequential()
        # ConvTranspose2d：上采样
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(256, 128, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_attention_' + str(layer_index), AttentionGate([128, 128], 128))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(256, 128, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(128, 128, kernel_size=self.block_size))
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True))
        self.decoder.add_module('decoder_l_'+str(layer_index), layer)

        layer_index = 0
        layer = nn.Sequential()
        # ConvTranspose2d：上采样
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(128, 64, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_attention_' + str(layer_index), AttentionGate([64, 64], 64))
        self.decoder.add_module('decoder_l_'+str(layer_index), layer)

        self.final_layers_lst = nn.ModuleList()
        # 理想情况下，有4个任务头：细胞检测、细胞分类、细胞子类别分类、细胞交叉K函数
        for i in range(self.n_heads):
            block = nn.Sequential()
            feat_subblock = nn.Sequential()
            pred_subblock = nn.Sequential()
            feat_subblock.add_module('final_block_'+str(i)+'_conv3_0', nn.Conv2d(128, 64, kernel_size=self.block_size))
            feat_subblock.add_module('final_block_'+str(i)+'_relu_0', nn.ReLU(inplace=True))
            feat_subblock.add_module('final_block_'+str(i)+'_conv3_1', nn.Conv2d(64, 64, kernel_size=self.block_size))
            feat_subblock.add_module('final_block_'+str(i)+'_relu_1', nn.ReLU(True))
            pred_subblock.add_module('final_block_'+str(i)+'_conv1_2', nn.Conv2d(64, self.head_classes[i], kernel_size=1))
            block.add_module('final_block_'+str(i) +'feat', feat_subblock)
            block.add_module('final_block_'+str(i) +'pred', pred_subblock)
            self.final_layers_lst.append(block)

        # self.final_final_block = nn.Sequential();
        # self.final_final_block.add_module('conv_final', nn.Conv2d(64*self.n_heads, self.n_classes, kernel_size=1));


        self._initialize_weights()

        self.zero_grad()

        print('self.encoder', self.encoder)
        print('self.bottleneck', self.bottleneck)
        print('self.decoder', self.decoder)

    '''
        init()只是初始化，forward()才是真正的功能函数
    '''
    def forward(self, x, feat_indx_list=[], feat_as_dict=False):
        """
            前向传播函数，计算模型的输出。
            
            参数：
            x：通过除以255进行归一化的输入图像
            feat_indx_list：包含不同模型块生成的特征的索引列表
                如果列表不为空，则将返回列表中列出的特征
                feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}
            feat_as_dict：如果feat_indx_list不为空，则以字典形式返回列表中指示的特征，
                其中键是特征索引标识符，值是特征

            返回：
            如果没有请求特征，则返回预测列表
            如果feat_as_dict为True，则返回包含特征的预测字典，否则返回特征的列表
        """

        feat = None  # 存储特征的变量
        feat_dict = {}  # 存储特征字典的变量
        feat_indx = 0  # 特征索引初始化
        attention_coefficients = []  # 存储 attention_coefficient 的列表
        # 跳跃连接实现
        # 1.编码器输出保存：在编码器的每一层（self.encoder）中，将输入x通过卷积层和激活函数处理，并将处理后的输出保存在encoder_out列表中。
        encoder_out = []  # 存储编码器输出的列表
        for l in self.encoder:     
            x = l(x)
            encoder_out.append(x)
        # 2.进入瓶颈层：将处理后的输入x传递给瓶颈层(self.bottleneck)。
        x = self.bottleneck(x)
        # 3.解码器处理：在编码器的每一层（self.decoder）中，首先通过l[0](x)获取当前层的输出x。
        j = len(self.decoder)
        for l in self.decoder:
        # for l, gate in zip(self.decoder, self.attention_gates):
            # 对输入x进行上采样操作
            # l[0](x)：l[0]表示该解码器层的第一个module，l[0](x)表示将输入x传递给该模块进行前向计算
            # l[0]：即第一个module，即上采样操作
            x = l[0](x)
            # g：下一层的解码器上采样的结果，用于Attention Gates的输入，即g，门控信号
            g = x
            j -= 1
            corresponding_layer_indx = j  # 需要连接的层

            # 4.跳跃连接实现：
            # 裁剪和连接
            if j >= 0:
                # 裁剪对应的编码器层的输出，使其大小与当前层的输出相同
                cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx],  x)
                # cropped：为同层的编码器，即xl
                # gate(g, x)
                # x_first = (2,512,32,32)
                # cropped = (2,512,64,64)
                gate_out, attention_coefficient = l[1](g, cropped)
                # 将裁剪后的编码器层的输出与当前层的输出在通道维度（1）上进行拼接
                x = torch.cat((gate_out, x), 1)
                attention_coefficients.append(attention_coefficient)
            for i in range(2, len(l)):
                # 对输入x执行剩下的子模块操作，即卷积、Relu、卷积、Relu
                x = l[i](x)

        # 检查是否将解码器特征返回到输出
        if feat_indx in feat_indx_list:
            if feat_as_dict:
                feat_dict[feat_indx] = x.detach().cpu().numpy()
            else:
                feat = x.detach().cpu().numpy()
        
        c = []  # 存储最终输出的列表
        f = None  # 存储特征的变量
        for layer in self.final_layers_lst:
            feat_indx += 1
            f1 = layer[0](x)  # 当前头的输出特征
            c.append(layer[1](f1))  # 当前头的输出预测
            if f is None:
                f = f1
            else:
                f = torch.cat((f1, f), 1)

            # 检查当前头的特征是否返回到输出
            if feat_indx in feat_indx_list:
                if feat_as_dict:
                    feat_dict[feat_indx] = f1.detach().cpu().numpy()
                else:
                    if feat is None:
                        feat = f1.detach().cpu().numpy()
                    else:
                        feat = np.concatenate((feat, f1.detach().cpu().numpy()), axis=1)

        # 如果没有请求特征，则只返回预测列表
        if len(feat_indx_list) == 0:
            # 训练过程中，返回的是这个
            return c, attention_coefficients

        # 返回包含请求特征的预测
        if feat_as_dict:
            return c, feat_dict, attention_coefficients
        return c, feat, attention_coefficients

    def _initialize_weights(self):
        for l in self.encoder:
            for layer in l:
                # 检查是否是卷积层
                if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                    if self.conv_init == 'normal':
                        torch.nn.init.normal_(layer.weight)
                    elif self.conv_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(layer.weight)
                    elif self.conv_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(layer.weight, gain=10)
                    elif self.conv_init == 'he':
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # 初始化瓶颈层权重
        for layer in self.bottleneck:
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                if self.conv_init == 'normal':
                    torch.nn.init.normal_(layer.weight)
                elif self.conv_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(layer.weight)
                elif self.conv_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(layer.weight, gain=10)
                elif self.conv_init == 'he':
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        # 初始化解码器权重
        for l in self.decoder:
            for layer in l:
                if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                    if self.conv_init == 'normal':
                        torch.nn.init.normal_(layer.weight)
                    elif self.conv_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(layer.weight)
                    elif self.conv_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(layer.weight, gain=10)
                    elif self.conv_init == 'he':
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        # 初始化最终层权重
        for layer in self.final_layers_lst:
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                if self.conv_init == 'normal':
                    torch.nn.init.normal_(layer.weight)
                elif self.conv_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(layer.weight)
                elif self.conv_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(layer.weight, gain=10)
                elif self.conv_init == 'he':
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

        # 从VGG-16预训练模型初始化编码器和瓶颈层
        vgg_model = models.vgg16(pretrained=True)
        fsd = collections.OrderedDict()
        i = 0
        # 将VGG模型的权重加载到编码器中
        for m in self.encoder.state_dict().items():
            temp_key = m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.encoder.load_state_dict(fsd)

        '''
            为了使用瓶颈层1024的通道数，只能暂时不将VGG模型的权重加载到瓶颈层中
            因为VGG16的权重最大只有512，所以不能匹配1024的瓶颈层
            不清楚取消瓶颈层的权重加载会不会影响最终结果
            
            2024.3.3
                恢复512通道，暂不考虑1024通道实验
        '''
        fsd = collections.OrderedDict()
        # 将VGG模型的权重加载到瓶颈层中
        for m in self.bottleneck.state_dict().items():
            temp_key = m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key] = list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.bottleneck.load_state_dict(fsd)

        # 删除VGG模型以释放内存
        # del vgg_model


class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()

        # 定义1x1卷积
        # conv_g:64×64×512（举例）
        self.conv_g = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # conv_x:64×64×512
        self.conv_x = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # conv_psi:64×64×1
        self.conv_psi = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

        # 定义Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        x_first = x  # 将x_first最后与out相乘

        # 将下一层解码器上采样的输出g通过1x1卷积处理
        g = self.conv_g(g)

        # 将同层解码器的输出x通过1x1卷积处理
        x = self.conv_x(x)

        # 将卷积后的结果相加
        out = g + x

        # 将相加后的结果通过ReLU激活函数处理
        out = self.relu(out)

        out = self.conv_psi(out)

        # attention_coefficient:注意系数
        attention_coefficient = self.sigmoid(out)

        out = x_first * attention_coefficient

        return out, attention_coefficient




