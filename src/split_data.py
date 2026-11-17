import os
import shutil
import random
from sklearn.model_selection import train_test_split

random.seed(42)

root = "../data/flower_photos"
dst_root = "../flower_data"
classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

for cls in classes:
    cls_path = os.path.join(root, cls)#类别路径
    img_path = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]#图片路径
    #7:2:1的比例划分训练，测试，预测集
    train,temp = train_test_split(img_path, test_size=0.3, random_state=42)
    val,test = train_test_split(temp, test_size=0.33, random_state=42)

    for split_name,img_list in zip(["train", "val", "test"],[train, val, test]):
        out_dir = os.path.join(dst_root, split_name,cls)
        os.makedirs(out_dir, exist_ok=True)
        for img_p in img_list:
            shutil.copy(img_p, out_dir)
print("✅ 数据集划分完成！")