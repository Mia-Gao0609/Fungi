import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from pathlib import Path

DATA = Path.home() / "fungi_data_33/train"
MODEL = Path.home() / "fungi_model_33.pth"

CLASS_NAMES = [
    "Agaricus_bisporus", "Amanita_muscaria", "Amanita_phalloides", "Amanita_verna",
    "Auricularia_auricula", "Boletus_edulis", "Calvatia_gigantea", "Cantharellus_cibarius",
    "Clitocybe_dealbata", "Coprinus_comatus", "Cortinarius_rubellus", "Dictyophora_duplicata",
    "Entoloma_sinuatum", "Flammulina_velutipes", "Galerina_marginata", "Grifola_frondosa",
    "Gyromitra_esculenta", "Gyromitra_infula", "Hericium_erinaceus", "Laetiporus_sulphureus",
    "Lactarius_deliciosus", "Lepiota_brunneoincarnata", "Morchella_esculenta", "Omphalotus_illudens",
    "Paxillus_involutus", "Phallus_indusiatus", "Pleurotus_ostreatus", "Russula_subnigricans",
    "Russula_virescens", "Termitomyces_eurhizus", "Tremella_fuciformis", "Tricholoma_matsutake",
    "Volvariella_volvacea"
]

NUM_CLASSES = len(CLASS_NAMES)

train_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ds = datasets.ImageFolder(DATA, transform=train_t)
train_n = int(0.8 * len(ds))
val_n = len(ds) - train_n
train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])
val_ds.dataset.transform = val_t

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=2)

print(f"Total: {len(ds)} images | Train: {train_n} | Val: {val_n}")

device = "cpu"  # 改成 CPU
print(f"💻 使用设备: {device} (CPU训练较慢，请耐心等待)")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.CrossEntropyLoss()
best_acc = 0.0

for epoch in range(5):  # 减少轮数，加快训练
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    scheduler.step()
    
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/5 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'class_names': CLASS_NAMES,
        }, MODEL)
        print(f"  ✅ Saved best model")

print(f"\nBest validation accuracy: {best_acc:.2f}%")
