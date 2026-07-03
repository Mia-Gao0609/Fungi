import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from pathlib import Path
import random

MODEL = Path.home() / "fungi_model_33.pth"
DATA = Path.home() / "fungi_data_33/train"

# 加载模型
checkpoint = torch.load(MODEL, map_location="cpu")
CLASS_NAMES = checkpoint['class_names']

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试5个随机类别
print("="*60)
print("模型诊断测试")
print("="*60)

for class_name in random.sample(CLASS_NAMES, 5):
    class_dir = DATA / class_name
    if not class_dir.exists():
        continue
    
    files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    if not files:
        continue
    
    # 随机选一张
    img_path = random.choice(files)
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(pixel_values=tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = probabilities.max(1)
        top3 = probabilities.topk(3)
    
    pred_class = CLASS_NAMES[pred.item()]
    print(f"\n真实类别: {class_name}")
    print(f"预测结果: {pred_class} (置信度: {conf.item()*100:.1f}%)")
    print("Top 3:")
    for i, (idx, prob) in enumerate(zip(top3.indices[0], top3.values[0])):
        print(f"  {i+1}. {CLASS_NAMES[idx]}: {prob.item()*100:.1f}%")

print(f"\n{'='*60}")
print("如果上面预测全是同一个类别，说明模型没训练好")
print("如果预测基本正确，说明是测试图片的问题")
print("="*60)
