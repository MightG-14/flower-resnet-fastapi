from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from model import create_resnet_model

app = FastAPI(title="花卉分类接口")

print("===== 开始加载模型 =====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("运行设备：", device)

model = create_resnet_model(num_classes=5, freeze_backbone=False)
print("✅ 模型结构创建完成")

model.load_state_dict(torch.load("../model.pth", map_location=device))
print("✅ 权重加载完成")

model = model.to(device)
model.eval()
print("✅ 模型加载就绪！")

class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 首页路由：GET请求，浏览器访问 127.0.0.1:8000 触发，返回HTML页面
@app.get("/", response_class=HTMLResponse)
async def home():
    html_page = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>花卉分类识别</title>
    <style>
        body {width:620px;margin:50px auto;font-family:Microsoft Yahei;}
        .card {border:1px solid #ddd;padding:24px;border-radius:10px;}
        #result {margin-top:24px;font-size:18px;color:#222;line-height:1.8;}
    </style>
</head>
<body>
    <div class="card">
        <h3>花卉图片识别系统</h3>
        <input type="file" id="imgFile" accept="image/*">
        <button onclick="uploadImage()">提交识别</button>
        <div id="result"></div>
    </div>

    <script>
        async function uploadImage(){
            const file = document.getElementById("imgFile").files[0];
            if(!file){alert("请先选择一张图片！");return;}
            const formData = new FormData();
            formData.append("file", file);
            const res = await fetch("/predict", {method:"POST", body:formData});
            const data = await res.json();
            document.getElementById("result").innerHTML = `
                <p>识别类别：${data.class_name}</p>
                <p>类别编号：${data.class_id}</p>
                <p>置信度：${data.confidence}</p>
            `
        }
    </script>
</body>
</html>
    """
    return html_page

# 预测接口路由：POST接口，接收上传图片，返回json结果
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()# 读取上传的二进制图片内容
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")# 二进制流打开图片，转RGB（防止png透明通道报错）
    img_tensor = transform(img).unsqueeze(0).to(device)
    # unsqueeze(0)：增加batch维度！模型输入形状 [batch,3,224,224]，单张图batch=1

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()# 取概率最大的索引（类别编号）
        confidence = prob[0][pred_idx].item()# 取出最大概率值

    return {
        "class_name": class_names[pred_idx],
        "class_id": pred_idx,
        "confidence": round(confidence, 4)
    }

