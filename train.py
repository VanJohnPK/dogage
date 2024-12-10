import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import re
import shutil

# 检查是否有 GPU 可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义自定义数据集
class DogAgeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []

        # 读取图像目录中的所有文件名
        img_files = os.listdir(img_dir)

        # 读取图像目录中的所有文件名
        img_files = os.listdir(img_dir)
        # print(f"Image files in directory: {img_files}")

        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_pattern = parts[0]
                age = int(parts[1])

                # 将文件名模式转换为正则表达式
                regex_pattern = re.escape(img_pattern).replace(r'\*', '.*')
                pattern = re.compile(f'^{regex_pattern}$')
                # print(f"Regex pattern: {regex_pattern}")

                # 在图像目录中找到匹配的文件
                matched_files = [img_name for img_name in img_files if pattern.match(img_name)]
                # print(f"Matched files: {matched_files}")
                if matched_files:
                    # 假设每个模式只匹配一个文件
                    self.img_labels.append((matched_files[0], age))

        print(f"Total matched files: {len(self.img_labels)}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, age = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, age


def main():
        
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    # annotations_file = 'annotations/test.txt'
    # image_dir = 'testset'
    annotations_file = 'annotations/train.txt'
    image_dir = 'trainset'
    val_file = 'annotations/val.txt'
    val_dir = 'valset'
    dataset = DogAgeDataset(annotations_file, image_dir, transform=transform)
    val_dataset = DogAgeDataset(val_file, val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)

    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    # 将模型移动到设备（GPU 或 CPU）
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # 训练模型
    num_epochs = 10
    print_freq = 1000
    best_prec1 = 1e6
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            # 将输入和标签移动到cuda
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # 使用定义的损失函数计算模型输出与目标值之间的误差
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_freq == 0:
                print(
                    "Epoch: [{0}][{1}/{2}]\t".format(
                        epoch,
                        i,
                        len(dataloader),
                    )
                )
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')
        prec1 = validate(model, val_loader)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        # print(prec1, best_prec1)
        print(" * best MAE {mae:.3f} ".format(mae=best_prec1))
        task = ""
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            task,
        )

    print('Finished Training')

# 用于验证模型在验证集上的表现
def validate(model, val_loader):
    print("begin test")

    model.eval()  # 将模型设置为评估模式，禁用 dropout 和 batch normalization 的训练行为。
    mae = 0  # 初始化平均绝对误差为 0

    for i, (img, target) in enumerate(val_loader):
        img = img.cuda()
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(val_loader)
    print(" * MAE {mae:.3f} ".format(mae=mae))

    return mae

def save_checkpoint(
    state, is_best, task_id, filename="checkpoint.pth.tar", save_dir="./model/"
):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(save_dir, task_id + "model_best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)


if __name__ == "__main__":
    main()