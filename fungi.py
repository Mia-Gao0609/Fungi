import torch,torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from transformers import ViTForImageClassification
from pathlib import Path

DATA=Path.home()/"fungi_data/train"
MODEL=Path.home()/"fungi_model.pth"

t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ds=datasets.ImageFolder(DATA,transform=t)
n=len(ds)
tr,va=torch.utils.data.random_split(ds,[int(0.8*n),n-int(0.8*n)])
tl=DataLoader(tr,batch_size=4,shuffle=True)
vl=DataLoader(va,batch_size=4)

device="cuda"
m=ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",num_labels=4,ignore_mismatched_sizes=True).to(device)
o=torch.optim.AdamW(m.parameters(),lr=1e-5)
c=nn.CrossEntropyLoss()

for e in range(3):
 m.train()
 for im,lb in tl:im,lb=im.to(device),lb.to(device);o.zero_grad();c(m(pixel_values=im).logits,lb).backward();o.step()
 m.eval();correct=0
 with torch.no_grad():
  for im,lb in vl:im,lb=im.to(device),lb.to(device);_,p=m(pixel_values=im).logits.max(1);correct+=p.eq(lb).sum().item()
 acc=100.*correct/len(va);print(f"Epoch {e+1}: {acc:.1f}%")

torch.save(m.state_dict(),MODEL)

import gradio as gr
m.load_state_dict(torch.load(MODEL,map_location=device));m.eval()

def predict(img):
 tensor=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(img).unsqueeze(0).to(device)
 with torch.no_grad():p=torch.nn.functional.softmax(m(pixel_values=tensor).logits,dim=1)
 conf,pred=p.max(1);return f"{ds.classes[pred.item()]} | {conf.item()*100:.1f}%"

gr.Interface(fn=predict,inputs=gr.Image(type="pil"),outputs="text",title="Fungi").launch(share=True)
