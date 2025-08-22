import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义分割头的结构
class SegmentationHead(nn.Module):
    def __init__(self, in_features, out_classes, scale_factor=2):
        super(SegmentationHead, self).__init__()
        # 1x1 卷积用于特征维度变换
        self.conv1x1 = nn.Conv2d(in_features, out_classes, kernel_size=3, padding=1)
        # 双线性上采样恢复分辨率
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 应用1x1卷积进行特征变换
        x = self.conv1x1(x)
        # 上采样到目标分辨率
        x = self.upsample(x)
        return x


import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.flatten = nn.Flatten()  # 扁平化层
        self.fc = nn.Linear(dim, num_classes)  # 全连接层

    def forward(self, x):
        x = self.global_avg_pool(x)  # 应用全局平均池化
        x = self.flatten(x)  # 扁平化特征图
        x = self.fc(x)  # 应用全连接层得到类别预测
        return x


# # 示例：初始化Swin Transformer的分类头
# # 假设特征维度为512，目标类别数为100
# classification_head = ClassificationHead(dim=512, num_classes=100)


# 示例初始化
# 假设从Swin Transformer提取的特征图通道数为512，目标分割类别数为20
seg_head = SegmentationHead(in_features=512, out_classes=100, scale_factor=4)
cls_head = ClassificationHead(dim=512, num_classes=100)
input_feature = torch.randn(1, 512, 64, 64)
output_seg = seg_head(input_feature)
print(output_seg.shape)
output_cls = cls_head(input_feature)
print(output_cls.shape)