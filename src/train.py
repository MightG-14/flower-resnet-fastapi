import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import create_resnet_model

batch_size = 16
lr = 1e-5
epochs = 5
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

train_loader, val_loader, class_names = load_data("../flower_data", batch_size=batch_size)
model = create_resnet_model(num_classes=num_classes,freeze_backbone=False)
model = model.to(device)
model.load_state_dict(torch.load("../model.pth",map_location=device))
print("权重加载成功")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    print(f"【Epoch {epoch + 1}/{epochs}】")
    print(f"训练 Loss:{train_loss:.4f}  训练 Acc:{train_acc:.4f}")
    print(f"验证 Loss:{val_loss:.4f}  验证 Acc:{val_acc:.4f}\n")

torch.save(model.state_dict(), "../model.pth")
print("完成")