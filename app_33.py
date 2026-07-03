import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import gradio as gr
from pathlib import Path

MODEL = Path.home() / "fungi_model_33.pth"

FUNGI_INFO = {
    "Agaricus_bisporus": {"cn": "双孢蘑菇", "edible": True, "note": "超市常见，可放心食用"},
    "Amanita_muscaria": {"cn": "毒蝇伞", "edible": False, "note": "有毒！含蝇蕈素"},
    "Amanita_phalloides": {"cn": "毒鹅膏", "edible": False, "note": "剧毒！致死率极高"},
    "Amanita_verna": {"cn": "白毒伞", "edible": False, "note": "极毒"},
    "Auricularia_auricula": {"cn": "黑木耳", "edible": True, "note": "凉拌炒食均可"},
    "Boletus_edulis": {"cn": "美味牛肝菌", "edible": True, "note": "需彻底煮熟"},
    "Calvatia_gigantea": {"cn": "大马勃", "edible": True, "note": "幼时可食"},
    "Cantharellus_cibarius": {"cn": "鸡油菌", "edible": True, "note": "炒食最佳"},
    "Clitocybe_dealbata": {"cn": "白霜杯伞", "edible": False, "note": "有毒"},
    "Coprinus_comatus": {"cn": "鸡腿菇", "edible": True, "note": "需新鲜食用"},
    "Cortinarius_rubellus": {"cn": "绯红丝膜菌", "edible": False, "note": "剧毒"},
    "Dictyophora_duplicata": {"cn": "短裙竹荪", "edible": True, "note": "需处理"},
    "Entoloma_sinuatum": {"cn": "毒粉褶菌", "edible": False, "note": "有毒"},
    "Flammulina_velutipes": {"cn": "金针菇", "edible": True, "note": "需煮熟"},
    "Galerina_marginata": {"cn": "盔孢伞", "edible": False, "note": "剧毒"},
    "Grifola_frondosa": {"cn": "灰树花", "edible": True, "note": "舞茸"},
    "Gyromitra_esculenta": {"cn": "鹿花菌", "edible": False, "note": "含鹿花菌素"},
    "Gyromitra_infula": {"cn": "褐鹿花菌", "edible": False, "note": "有毒"},
    "Hericium_erinaceus": {"cn": "猴头菇", "edible": True, "note": "养胃佳品"},
    "Laetiporus_sulphureus": {"cn": "硫色绚孔菌", "edible": True, "note": "鸡肉味"},
    "Lactarius_deliciosus": {"cn": "松乳菇", "edible": True, "note": "需煮熟"},
    "Lepiota_brunneoincarnata": {"cn": "褐鳞小伞", "edible": False, "note": "剧毒"},
    "Morchella_esculenta": {"cn": "羊肚菌", "edible": True, "note": "必须煮熟"},
    "Omphalotus_illudens": {"cn": "发光脐菇", "edible": False, "note": "有毒"},
    "Paxillus_involutus": {"cn": "卷边桩菇", "edible": False, "note": "有毒"},
    "Phallus_indusiatus": {"cn": "竹荪", "edible": True, "note": "名贵菌类"},
    "Pleurotus_ostreatus": {"cn": "平菇", "edible": True, "note": "常见食用菌"},
    "Russula_subnigricans": {"cn": "亚稀褶黑菇", "edible": False, "note": "有毒"},
    "Russula_virescens": {"cn": "变绿红菇", "edible": True, "note": "需确认品种"},
    "Termitomyces_eurhizus": {"cn": "鸡枞菌", "edible": True, "note": "白蚁共生"},
    "Tremella_fuciformis": {"cn": "银耳", "edible": True, "note": "滋补佳品"},
    "Tricholoma_matsutake": {"cn": "松茸", "edible": True, "note": "名贵食用菌"},
    "Volvariella_volvacea": {"cn": "草菇", "edible": True, "note": "高温菇"},
}

device = "cpu"
checkpoint = torch.load(MODEL, map_location=device)
CLASS_NAMES = checkpoint['class_names']

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict(img):
    if img is None:
        return None, "请先上传图片"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(pixel_values=tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    pred_idx = predicted.item()
    pred_class = CLASS_NAMES[pred_idx]
    conf = confidence.item()
    
    top5_probs, top5_indices = probabilities.topk(min(5, len(CLASS_NAMES)))
    top5 = [(CLASS_NAMES[i], top5_probs[0][j].item()) for j, i in enumerate(top5_indices[0])]
    
    info = FUNGI_INFO.get(pred_class, {"cn": "未知", "edible": False, "note": "暂无信息"})
    color = "#4CAF50" if info["edible"] else "#F44336"
    
    html = f"""
    <div style="padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
        <h2>🍄 识别结果</h2>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; color: #333; margin: 15px 0;">
            <div style="font-size: 28px; font-weight: bold; color: {color};">{pred_class}</div>
            <div style="font-size: 22px; color: #666; margin-top: 8px;">{info['cn']}</div>
            <div style="margin-top: 12px; font-size: 18px;">置信度: <b style="color: {color};">{conf*100:.1f}%</b></div>
        </div>
        <div style="background: {color}30; padding: 15px; border-radius: 10px; border: 2px solid {color};">
            <div style="font-size: 24px; font-weight: bold; color: {color};">{'✅ 可食用' if info['edible'] else '❌ 有毒'}</div>
            <div style="font-size: 16px; margin-top: 8px;">{info['note']}</div>
        </div>
        <div style="margin-top: 20px;">
            <h4>🔍 Top 5 候选:</h4>
    """
    
    for cls, prob in top5:
        ci = FUNGI_INFO.get(cls, {"cn": "?", "edible": True})
        cc = "#4CAF50" if ci["edible"] else "#F44336"
        html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{cls} ({ci['cn']})</span>
                    <span style="color: {cc}; font-weight: bold;">{prob*100:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 24px;">
                    <div style="background: {cc}; width: {int(prob*100)}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
        """
    
    html += "</div></div>"
    return img, html

with gr.Blocks(title="🍄 Fungi Classifier 33", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 48px;">🍄 Fungi Classifier</h1>
        <h3 style="color: #666;">33类真菌AI识别系统</h3>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 上传蘑菇图片")
            image_input = gr.Image(type="pil", height=450)
            btn = gr.Button("🔍 开始识别", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 提示
            - 上传清晰的蘑菇照片
            - 包含菌盖、菌柄效果更好
            
            ### ⚠️ 安全声明
            **本系统仅供参考，请勿仅凭AI判断食用野生蘑菇！**
            """)
        
        with gr.Column():
            gr.Markdown("### 📊 识别结果")
            image_output = gr.Image(type="pil", height=350)
            result_output = gr.HTML()
    
    btn.click(predict, image_input, [image_output, result_output])

demo.launch(share=True, server_port=7860)
