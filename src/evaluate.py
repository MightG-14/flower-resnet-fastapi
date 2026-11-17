import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_data
from model import create_resnet_model

batch_size = 16
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"评估使用设备：{device}")

train_loader, val_loader, class_names = load_data("../flower_data", batch_size=batch_size)

model = create_resnet_model(num_classes = num_classes,freeze_backbone=False)
model = model.to(device)

model.load_state_dict(torch.load("../model.pth",map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print("="*50)
print("分类报告（精确率 / 召回率 / F1）")
print(classification_report(all_labels,all_preds,target_names=class_names))
print("="*50)

#计算混淆矩阵
cm = confusion_matrix(all_labels,all_preds)
print(f"混淆矩阵数值：{cm}")

#绘图
plt.figure(figsize=(6,6))
# 画热力图，annot=True 显示格子里数字，fmt='d' 是整数
sns.heatmap(cm,annot=True,fmt="d",cmap = 'Blues',
            xticklabels=class_names,yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../confusion_matrix.png",dpi = 300)
plt.show()



