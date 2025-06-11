import torch
from torch import nn

# 定义一个名为 alex 的类，继承自 nn.Module，用于构建神经网络模型
class alex(nn.Module):
    def __init__(self, num_class=10):
        super(alex, self).__init__()
        # 初始化模型的各个层，使用 Sequential 容器来组织这些层
        self.model = nn.Sequential(
            # 第一层卷积，输入通道数为3（RGB图像），输出通道数为48，卷积核大小为5，步长为4
            nn.Conv2d(3, 48, kernel_size=5, stride=4),
            # 第一层池化，使用最大池化，池化窗口大小为3，步长为2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二层卷积，输入通道数为48，输出通道数为128，卷积核大小为3
            nn.Conv2d(48, 128, kernel_size=3),
            # 第二层池化，使用最大池化，池化窗口大小为2
            nn.MaxPool2d(kernel_size=2),
            # 第三层卷积，输入通道数为128，输出通道数为192，卷积核大小为3
            nn.Conv2d(128, 192, kernel_size=3),
            # 第四层卷积，输入通道数为192，输出通道数为192，卷积核大小为3
            nn.Conv2d(192, 192, kernel_size=3),
            # 第五层卷积，输入通道数为192，输出通道数为128，卷积核大小为3
            nn.Conv2d(192, 128, kernel_size=3),
            # 第三层池化，使用最大池化，池化窗口大小为2
            nn.MaxPool2d(kernel_size=2),
            # 将多维的卷积层输出展平成一维，以便输入到全连接层
            nn.Flatten(),
            # 第一层全连接，输入特征数为128*3*3（根据前面的卷积和池化层计算得出），输出特征数为2048
            nn.Linear(128 * 3 * 3, 2048),
            # 第二层全连接，输入特征数为2048，输出特征数为1024
            nn.Linear(2048, 1024),
            # 第三层全连接，输入特征数为1024，输出特征数为分类类别数（num_class）
            nn.Linear(1024, num_class),
        )

        # 定义分类器部分，使用 Sequential 容器来组织这些层
        self.classifier = nn.Sequential(
            # Dropout层，用于防止过拟合，丢弃率为0.5
            nn.Dropout(p=0.5),
            # 第一层全连接，输入特征数为256*6*6（根据前面的卷积和池化层计算得出），输出特征数为4096
            nn.Linear(256 * 6 * 6, 4096),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 再次使用Dropout层，丢弃率为0.5
            nn.Dropout(p=0.5),
            # 第二层全连接，输入特征数为4096，输出特征数为4096
            nn.Linear(4096, 4096),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 第三层全连接，输入特征数为4096，输出特征数为分类类别数（num_class）
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        # 前向传播函数，定义了数据如何通过网络
        y = self.model(x)
        # 这里注释掉了一些代码，可能是为了调试或测试不同网络结构
        # y = F.interpolate(
        #     x,
        #     size=(224, 224),
        #     mode='bilinear',
        #     align_corners=False
        # )
        # y = self.features(y)
        # y = torch.flatten(y, 1)
        # y = self.classifier(y)

        return y

if __name__ == '__main__':
    # 创建一个随机输入张量，模拟一个批次的图像数据，批次大小为1，图像尺寸为3x224x224
    x = torch.randn(1, 3, 224, 224)
    # 实例化模型
    alexnet = alex()
    # 将输入数据通过网络，得到输出
    y = alexnet(x)
    # 打印输出的形状，以验证网络结构是否正确
    print(y.shape)

    # 不同输入尺寸的网络输出
    # model = alex()
    # x = torch.randm(1, 3, 32, 32)
    # y = model(x)
    # print(y.shape)
