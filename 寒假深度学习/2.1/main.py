from clss import ResNet_CNN
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

epochs = 10 #训练轮数
batch_size_train = 64   #训练集批量大小
batch_size_test = 1000  #测试集批量大小
learning_rate = 0.008    #学习率，即步长
log_interval = 10   #每隔多少个batch打印训练日志
random_seed = 1     # 随机种子
torch.manual_seed(random_seed)  #设置随机种子


# 训练模型并记录准确度和损失
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []

# 加载MNIST数据集并创建数据加载器
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_test, shuffle=True)


# 判断是否可以使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型、损失函数和优化器
model = ResNet_CNN().to(device)     #实例化ResNet_CNN网络并移动到设备上
loss_f = nn.CrossEntropyLoss()      #交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #Adam优化器,性能优于SGD


# 训练函数
def train(epochs):
    for epoch in range(epochs):     #循环遍历每个训练轮次
        model.train()       #将模型设置为训练模式
        train_loss = 0      #跟踪损失值
        correct_train = 0   #正确判断的个数
        total_train = 0     #总训练个数
        for batch_idx, (data, target) in enumerate(train_loader):   #遍历训练数据集，data是输入图像数据，target是对应的标签
            data, target = data.to(device), target.to(device)   #输入数据和标签移动到GPU上
            optimizer.zero_grad()   #清除之前的梯度，以避免梯度累积
            output = model(data)    #将输入数据传递给模型并得到输出
            loss = loss_f(output, target)   #计算模型输出与实际标签之间的损失
            loss.backward()     #误差反向传播计算梯度
            optimizer.step()    #根据计算的梯度更新参数
            train_loss += loss.item()       #计算训练集损失，正确率
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))
                if epoch == 9:
                    train_loss_list.append(loss.item())

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct_train / total_train
        train_accuracy_list.append(train_accuracy)

        test_accuracy, correct_test = test()
        test_accuracy_list.append(test_accuracy)

        print('\nEpoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, train_loss, correct_train, total_train, train_accuracy))
        print('Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct_test, len(test_loader.dataset), test_accuracy))

    #保存模型
    example_input = torch.rand(1, 1, 28, 28).to(device)     #模型输入为28*28的图像
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, "traced_model1.pt")

    # 绘制准确度图表
    plt.plot(train_accuracy_list, label='Train Accuracy')
    plt.plot(test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()

    # 绘制最后一次epoch损失图表
    plt.plot(train_loss_list, label='Train Loss')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

#测试函数
def test():
    model.eval()    #禁用 Dropout 和 Batch Normalization 等层的训练模式，设置模型为测试模式
    correct = 0
    with torch.no_grad():   # 上下文管理器，以避免进行梯度计算，从而加快计算速度
        for data, target in test_loader:    #遍历测试数据集，data是输入图像数据，target是对应的标签
            data, target = data.to(device), target.to(device)   #将输入数据和标签移动到GPU上
            output = model(data)    #将输入数据传递给模型并得到输出
            pred = output.argmax(dim=1, keepdim=True)   #找到每个输入数据的预测标签
            correct += pred.eq(target.view_as(pred)).sum().item()   #将正确预测的数量添加到 correct 变量中
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy, correct


if __name__ == '__main__':
    train(epochs)
