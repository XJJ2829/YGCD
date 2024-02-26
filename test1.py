import torch
import EffNet
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # 其他的数据预处理操作
])

# 加载测试集数据
test_data = DatasetFolder("E:/first/1/Test/Data/ChangSha/test100", loader=torchvision.datasets.folder.default_loader, extensions=".jpg",
                          transform=transform)

test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 加载模型
model = torch.load("E:/first/1/Test/Models/effnet_newChangSha.pth")
model.eval()

# 在测试集上进行预测
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        # 前向传播
        outputs = model(images)

        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        # 统计预测结果与标签一致的数量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算准确率
accuracy = correct / total
print(f"测试集的准确率为: {accuracy}")