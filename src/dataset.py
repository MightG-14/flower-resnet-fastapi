import torch
#from sklearn import datasets
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

def get_transform():
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    return train_transform, val_transform

def load_data(root_path,batch_size = 16):
    train_transform, val_transform = get_transform()
    train_dataset = datasets.ImageFolder(root=root_path+"/train",transform=train_transform)
    val_dataset = datasets.ImageFolder(root=root_path+"/val",transform=val_transform)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    class_names = train_dataset.classes
    print(class_names)
    return train_loader, val_loader ,class_names

if __name__ == "__main__":
    train_loader, val_loader,class_names = load_data("../flower_data")
    print("训练集批次数量：", len(train_loader))
    print("验证集批次数量：", len(val_loader))
    train_loader, val_loader, cls = load_data("../flower_data", batch_size=8)
    print("加载成功！")