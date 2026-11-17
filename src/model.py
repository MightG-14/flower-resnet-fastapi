import torch.nn as nn
from torchvision import models


def create_resnet_model(num_classes=5, freeze_backbone=True):
    # 1. 创建resnet50，加载imagenet预训练权重
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # =========【重点：不管冻结与否，先替换分类头！放到if外面！】=========
    # 获取原来fc层的输入维度：resnet50是2048
    in_features = model.fc.in_features
    # 重新构造线性层，输出改为我们的5类
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    # 2. 冻结主干逻辑：只在freeze_backbone=True时冻结主干
    if freeze_backbone:
        # 遍历所有参数，全部设置为不需要计算梯度（冻结）
        for param in model.parameters():
            param.requires_grad = False
        # 但是！fc层是我们刚刚新建的，我们要训练它，所以单独放开fc的梯度
        model.fc.requires_grad_(True)

    # freeze_backbone=False的时候：上面if不执行，所有参数默认requires_grad=True，全部解冻微调
    return model


if __name__ == "__main__":
    # 测试1：冻结主干模式（第一轮训练，只训练fc）
    print("===== 冻结主干模式（freeze_backbone=True）=====")
    model_freeze = create_resnet_model(num_classes=5, freeze_backbone=True)
    for name, param in model_freeze.named_parameters():
        if param.requires_grad:
            print("可训练：", name)

    print("\n===== 解冻微调模式（freeze_backbone=False）=====")
    # 测试2：解冻模式（第二轮微调，全部参数可训练）
    model_unfreeze = create_resnet_model(num_classes=5, freeze_backbone=False)
    for name, param in model_unfreeze.named_parameters():
        if param.requires_grad:
            print("可训练：", name)



