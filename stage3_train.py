import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from pathlib import Path

DATA = Path.home() / "fungi_data_100/train"
MODEL = Path.home() / "fungi_model_best.pth"

CLASS_NAMES = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
print(f"类别: {NUM_CLASSES}")

train_t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_t = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = datasets.ImageFolder(DATA, transform=train_t)
train_n = int(0.85*len(ds))
val_n = len(ds)-train_n
train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])
val_ds.dataset.transform = val_t

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=2)

print(f"Total: {len(ds)} | Train: {train_n} | Val: {val_n}")

device = "cpu"
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
best_acc = 0

# 冻结10轮
print("\n=== 冻结主干（10轮）===")
for param in model.vit.parameters():
    param.requires_grad = False

for epoch in range(10):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        criterion(outputs, labels).backward()
        optimizer.step()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    train_acc = 100.*correct/total
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(pixel_values=images).logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    val_acc = 100.*correct/total
    
    print(f"Epoch {epoch+1}/10 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'classes': CLASS_NAMES}, MODEL)
        print(f"  ✅ Best: {val_acc:.2f}%")

# 微调10轮
print("\n=== 解冻微调（10轮）===")
for param in model.vit.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

for epoch in range(10):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        criterion(outputs, labels).backward()
        optimizer.step()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    train_acc = 100.*correct/total
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(pixel_values=images).logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    val_acc = 100.*correct/total
    
    print(f"Fine-tune {epoch+1}/10 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc, 'classes': CLASS_NAMES}, MODEL)
        print(f"  ✅ Best: {val_acc:.2f}%")

print(f"\n🎉 最终最佳: {best_acc:.2f}%")
