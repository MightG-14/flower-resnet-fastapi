# 花卉五分类系统｜ResNet50 迁移学习 + FastAPI部署
> 基于PyTorch实现ResNet50迁移学习图像分类
> 完成「数据集划分→训练微调→指标评估→Web服务部署」完整CV项目闭环。

## 项目简介
数据集：Kaggle Flower Photos 花卉数据集，共5类花卉：daisy雏菊、dandelion蒲公英、roses玫瑰、sunflowers向日葵、tulips郁金香。
技术栈：Python + PyTorch + torchvision + ResNet50 + FastAPI + Matplotlib + Seaborn + Scikit-learn
功能：
1. 数据集自动划分训练集/验证集
2. 两阶段迁移学习：冻结主干训练 + 解冻微调提升精度
3. 模型评估：混淆矩阵、精确率、召回率、F1分数
4. FastAPI Web接口，支持图片上传推理
5. 内嵌极简前端页面，浏览器可视化识别

## 📁 项目目录
flower_photo/
├── data/
│   └── flower_photos/        # Kaggle 原始数据集
├── flower_data/              # 划分后的训练、验证集
├── src/
│   ├── split_data.py         # 数据集划分脚本
│   ├── dataset.py            # 数据增强、DataLoader
│   ├── model.py              # ResNet50 迁移模型构建
│   ├── train.py              # 训练 + 解冻微调
│   ├── evaluate.py           # 评估指标、混淆矩阵
│   └── api.py                # FastAPI 推理服务
└── model.pth                 # 训练完成权重文件

## 🛠️ 环境安装
````
pip install torch torchvision
pip install fastapi uvicorn
pip install pillow matplotlib seaborn scikit-learn
````

## 🚀 项目运行步骤
python split_data.py #数据集划分
python train.py      #模型训练
训练分为两个阶段：
1. **冻结主干训练**：冻结 ResNet50 主干网络全部参数，只训练最后的全连接分类头。利用 ImageNet 预训练学到的通用图像特征，适合小数据集，快速收敛，降低过拟合风险
2. **解冻微调训练**：放开主干网络所有参数，使用很小的学习率，在花卉数据集上微调全部权重，进一步提升分类精度
训练结束后，权重自动保存为项目根目录 `model.pth`

python evaluate.py   #模型效果评估
uvicorn api:app      #启动 Web 推理服务
返回 JSON 示例：
{
    "class_name": "dandelion",
    "class_id": 1,
    "confidence": 0.997
}

