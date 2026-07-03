import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import gradio as gr
from pathlib import Path

MODEL = Path.home() / "fungi_model_best.pth"

# 界面文本
UI = {
    "zh": {
        "title": "🍄 真菌识别系统",
        "subtitle": "33类真菌AI识别 - 智能判断可食用性",
        "upload": "📤 上传蘑菇图片",
        "recognize": "🔍 开始识别",
        "result": "📊 识别结果",
        "edible": "✅ 可食用",
        "poisonous": "❌ 有毒",
        "how": "🍽️ 食用方法",
        "warning": "⚠️ 注意事项",
        "top5": "🔍 Top 5 候选",
        "tips": "💡 提示",
        "safety": "⚠️ 安全声明",
        "safety_text": "本系统仅供参考，请勿仅凭AI判断食用野生蘑菇！有毒蘑菇与可食用蘑菇外观相似，误食可能致命。",
        "lang_label": "语言 / Language"
    },
    "en": {
        "title": "🍄 Fungi Classifier",
        "subtitle": "33-class AI Fungi Recognition - Edibility Assessment",
        "upload": "📤 Upload Mushroom Photo",
        "recognize": "🔍 Recognize",
        "result": "📊 Result",
        "edible": "✅ Edible",
        "poisonous": "❌ Poisonous",
        "how": "🍽️ How to Eat",
        "warning": "⚠️ Warning",
        "top5": "🔍 Top 5 Candidates",
        "tips": "💡 Tips",
        "safety": "⚠️ Safety Disclaimer",
        "safety_text": "This system is for reference only. Do not rely solely on AI for wild mushroom identification! Poisonous mushrooms can resemble edible ones, and ingestion can be fatal.",
        "lang_label": "Language / 语言"
    }
}

# 真菌信息（中英双语）
FUNGI_INFO = {
    "Agaricus_bisporus": {
        "zh": {"name": "双孢蘑菇", "how": "炒食、煮汤、烧烤均可", "warning": "超市常见，放心食用"},
        "en": {"name": "Button Mushroom", "how": "Stir-fry, soup, or grill", "warning": "Common in supermarkets, safe to eat"}
    },
    "Amanita_muscaria": {
        "zh": {"name": "毒蝇伞", "how": "不可食用", "warning": "剧毒！含蝇蕈素，误食可致幻觉、昏迷甚至死亡"},
        "en": {"name": "Fly Agaric", "how": "Not edible", "warning": "Highly toxic! Contains muscimol, can cause hallucinations, coma, or death"}
    },
    "Amanita_phalloides": {
        "zh": {"name": "毒鹅膏", "how": "不可食用", "warning": "极毒！致死率极高，外观与可食用蘑菇相似，极易误食"},
        "en": {"name": "Death Cap", "how": "Not edible", "warning": "Extremely toxic! High fatality rate, resembles edible mushrooms"}
    },
    "Amanita_verna": {
        "zh": {"name": "白毒伞", "how": "不可食用", "warning": "极毒！与可食用蘑菇外观相似，误食致命"},
        "en": {"name": "Destroying Angel", "how": "Not edible", "warning": "Extremely toxic! Resembles edible mushrooms, ingestion is fatal"}
    },
    "Auricularia_auricula": {
        "zh": {"name": "黑木耳", "how": "凉拌、炒食、煮汤", "warning": "需充分泡发后食用"},
        "en": {"name": "Wood Ear", "how": "Cold dish, stir-fry, or soup", "warning": "Soak thoroughly before cooking"}
    },
    "Boletus_edulis": {
        "zh": {"name": "美味牛肝菌", "how": "炒食、炖汤、油煎", "warning": "必须彻底煮熟，生吃有毒"},
        "en": {"name": "Porcini", "how": "Stir-fry, soup, or pan-fry", "warning": "Must be thoroughly cooked, raw consumption is toxic"}
    },
    "Calvatia_gigantea": {
        "zh": {"name": "大马勃", "how": "嫩时炒食、煮汤", "warning": "幼嫩时可食，成熟后不可食用"},
        "en": {"name": "Giant Puffball", "how": "Stir-fry or soup when young", "warning": "Edible when young, inedible when mature"}
    },
    "Cantharellus_cibarius": {
        "zh": {"name": "鸡油菌", "how": "炒食、炖汤", "warning": "香气浓郁，炒食最佳"},
        "en": {"name": "Chanterelle", "how": "Stir-fry or soup", "warning": "Fragrant, best when stir-fried"}
    },
    "Clitocybe_dealbata": {
        "zh": {"name": "白霜杯伞", "how": "不可食用", "warning": "有毒！误食可引起中毒症状"},
        "en": {"name": "Sweating Fairy", "how": "Not edible", "warning": "Toxic! Ingestion can cause poisoning symptoms"}
    },
    "Coprinus_comatus": {
        "zh": {"name": "鸡腿菇", "how": "炒食、煮汤", "warning": "需新鲜食用，老化后不可食用"},
        "en": {"name": "Shaggy Mane", "how": "Stir-fry or soup", "warning": "Must be fresh, inedible when aged"}
    },
    "Cortinarius_rubellus": {
        "zh": {"name": "绯红丝膜菌", "how": "不可食用", "warning": "剧毒！含肾毒素，误食可致肾衰竭"},
        "en": {"name": "Deadly Webcap", "how": "Not edible", "warning": "Highly toxic! Contains nephrotoxins, can cause kidney failure"}
    },
    "Dictyophora_duplicata": {
        "zh": {"name": "短裙竹荪", "how": "煮汤、火锅", "warning": "需处理干净后食用"},
        "en": {"name": "Veiled Lady", "how": "Soup or hot pot", "warning": "Clean thoroughly before cooking"}
    },
    "Entoloma_sinuatum": {
        "zh": {"name": "毒粉褶菌", "how": "不可食用", "warning": "有毒！外观与可食用粉褶菌相似"},
        "en": {"name": "Livid Entoloma", "how": "Not edible", "warning": "Toxic! Resembles edible Entoloma species"}
    },
    "Flammulina_velutipes": {
        "zh": {"name": "金针菇", "how": "火锅、炒食、煮汤", "warning": "需煮熟后食用"},
        "en": {"name": "Enoki Mushroom", "how": "Hot pot, stir-fry, or soup", "warning": "Must be cooked before eating"}
    },
    "Galerina_marginata": {
        "zh": {"name": "盔孢伞", "how": "不可食用", "warning": "剧毒！含鹅膏毒素，致死率高"},
        "en": {"name": "Deadly Galerina", "how": "Not edible", "warning": "Highly toxic! Contains amatoxins, high fatality rate"}
    },
    "Grifola_frondosa": {
        "zh": {"name": "灰树花", "how": "炒食、炖汤、煮粥", "warning": "又名舞茸，需煮熟后食用"},
        "en": {"name": "Maitake", "how": "Stir-fry, soup, or porridge", "warning": "Also known as Maitake, must be cooked"}
    },
    "Gyromitra_esculenta": {
        "zh": {"name": "鹿花菌", "how": "不可食用", "warning": "有毒！含鹿花菌素，误食可致中毒"},
        "en": {"name": "False Morel", "how": "Not edible", "warning": "Toxic! Contains gyromitrin, can cause poisoning"}
    },
    "Gyromitra_infula": {
        "zh": {"name": "褐鹿花菌", "how": "不可食用", "warning": "有毒！与鹿花菌同属，不可食用"},
        "en": {"name": "Hooded False Morel", "how": "Not edible", "warning": "Toxic! Related to False Morel, not edible"}
    },
    "Hericium_erinaceus": {
        "zh": {"name": "猴头菇", "how": "炖汤、煮粥、炒食", "warning": "养胃佳品，需充分煮熟"},
        "en": {"name": "Lion's Mane", "how": "Soup, porridge, or stir-fry", "warning": "Good for stomach health, must be thoroughly cooked"}
    },
    "Laetiporus_sulphureus": {
        "zh": {"name": "硫色绚孔菌", "how": "炒食、炖汤", "warning": "鸡肉味，需煮熟后食用"},
        "en": {"name": "Chicken of the Woods", "how": "Stir-fry or soup", "warning": "Tastes like chicken, must be cooked"}
    },
    "Lactarius_deliciosus": {
        "zh": {"name": "松乳菇", "how": "炒食、炖汤", "warning": "需煮熟后食用，有松香味"},
        "en": {"name": "Saffron Milk Cap", "how": "Stir-fry or soup", "warning": "Must be cooked, has a pine aroma"}
    },
    "Lepiota_brunneoincarnata": {
        "zh": {"name": "褐鳞小伞", "how": "不可食用", "warning": "剧毒！含鹅膏毒素，致死率高"},
        "en": {"name": "Deadly Dapperling", "how": "Not edible", "warning": "Highly toxic! Contains amatoxins, high fatality rate"}
    },
    "Morchella_esculenta": {
        "zh": {"name": "羊肚菌", "how": "炖汤、炒食", "warning": "必须彻底煮熟，不可生吃"},
        "en": {"name": "Morel", "how": "Soup or stir-fry", "warning": "Must be thoroughly cooked, never eat raw"}
    },
    "Omphalotus_illudens": {
        "zh": {"name": "发光脐菇", "how": "不可食用", "warning": "有毒！夜间发光，误食可引起中毒"},
        "en": {"name": "Jack-o'-Lantern", "how": "Not edible", "warning": "Toxic! Bioluminescent at night, causes poisoning"}
    },
    "Paxillus_involutus": {
        "zh": {"name": "卷边桩菇", "how": "不可食用", "warning": "有毒！可引起溶血反应"},
        "en": {"name": "Brown Roll-Rim", "how": "Not edible", "warning": "Toxic! Can cause hemolysis"}
    },
    "Phallus_indusiatus": {
        "zh": {"name": "竹荪", "how": "炖汤、火锅", "warning": "名贵菌类，需处理干净后食用"},
        "en": {"name": "Bamboo Fungus", "how": "Soup or hot pot", "warning": "Premium fungus, clean thoroughly before cooking"}
    },
    "Pleurotus_ostreatus": {
        "zh": {"name": "平菇", "how": "炒食、煮汤、火锅", "warning": "常见食用菌，需煮熟后食用"},
        "en": {"name": "Oyster Mushroom", "how": "Stir-fry, soup, or hot pot", "warning": "Common edible mushroom, must be cooked"}
    },
    "Russula_subnigricans": {
        "zh": {"name": "亚稀褶黑菇", "how": "不可食用", "warning": "有毒！外观与可食用红菇相似"},
        "en": {"name": "Blackening Russula", "how": "Not edible", "warning": "Toxic! Resembles edible Russula species"}
    },
    "Russula_virescens": {
        "zh": {"name": "变绿红菇", "how": "炒食", "warning": "需确认品种，部分红菇有毒"},
        "en": {"name": "Green Cracking Russula", "how": "Stir-fry", "warning": "Verify species, some Russula are toxic"}
    },
    "Termitomyces_eurhizus": {
        "zh": {"name": "鸡枞菌", "how": "炒食、炖汤、油炸", "warning": "白蚁共生菌，需煮熟后食用"},
        "en": {"name": "Termite Mushroom", "how": "Stir-fry, soup, or deep-fry", "warning": "Termite symbiotic fungus, must be cooked"}
    },
    "Tremella_fuciformis": {
        "zh": {"name": "银耳", "how": "炖汤、煮粥、凉拌", "warning": "滋补佳品，需充分泡发后食用"},
        "en": {"name": "White Fungus", "how": "Soup, porridge, or cold dish", "warning": "Nutritious tonic, soak thoroughly before use"}
    },
    "Tricholoma_matsutake": {
        "zh": {"name": "松茸", "how": "烤食、炖汤、炒食", "warning": "名贵食用菌，需轻处理保留香气"},
        "en": {"name": "Matsutake", "how": "Grill, soup, or stir-fry", "warning": "Premium fungus, handle gently to preserve aroma"}
    },
    "Volvariella_volvacea": {
        "zh": {"name": "草菇", "how": "炒食、煮汤", "warning": "高温菇类，需煮熟后食用"},
        "en": {"name": "Straw Mushroom", "how": "Stir-fry or soup", "warning": "Heat-loving fungus, must be cooked"}
    },
}

checkpoint = torch.load(MODEL, map_location="cpu")
CLASS_NAMES = checkpoint['classes']

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict(img, lang):
    if img is None:
        return None, "Please upload an image / 请先上传图片"
    
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = t(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(pixel_values=tensor).logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    
    pred_class = CLASS_NAMES[pred.item()]
    confidence = conf.item()
    top5 = [(CLASS_NAMES[i], probs[0][j].item()) for j, i in enumerate(probs.topk(5)[1][0])]
    
    info = FUNGI_INFO.get(pred_class, {
        "zh": {"name": "未知", "how": "未知", "warning": "暂无信息"},
        "en": {"name": "Unknown", "how": "Unknown", "warning": "No information"}
    })
    
    # 根据语言生成输出
    if lang == "中文":
        ui = UI["zh"]
        i = info["zh"]
        name = i["name"]
        how = i["how"]
        warning = i["warning"]
        edible_text = ui["edible"] if info["zh"]["how"] != "不可食用" else ui["poisonous"]
        color = "#4CAF50" if info["zh"]["how"] != "不可食用" else "#F44336"
    elif lang == "English":
        ui = UI["en"]
        i = info["en"]
        name = i["name"]
        how = i["how"]
        warning = i["warning"]
        edible_text = ui["edible"] if info["en"]["how"] != "Not edible" else ui["poisonous"]
        color = "#4CAF50" if info["en"]["how"] != "Not edible" else "#F44336"
    else:  # 双语
        ui = UI["zh"]
        i_zh = info["zh"]
        i_en = info["en"]
        name = f"{i_zh['name']} / {i_en['name']}"
        how = f"{i_zh['how']} / {i_en['how']}"
        warning = f"{i_zh['warning']} / {i_en['warning']}"
        edible_text = "✅ 可食用 / Edible" if info["zh"]["how"] != "不可食用" else "❌ 有毒 / Poisonous"
        color = "#4CAF50" if info["zh"]["how"] != "不可食用" else "#F44336"
    
    html = f"""
    <div style="padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
        <h2>{ui['result']}</h2>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; color: #333; margin: 15px 0;">
            <div style="font-size: 28px; font-weight: bold; color: {color};">{pred_class}</div>
            <div style="font-size: 22px; color: #666; margin-top: 8px;">{name}</div>
            <div style="margin-top: 12px; font-size: 18px;">{ui['result']} Confidence: <b style="color: {color};">{confidence*100:.1f}%</b></div>
        </div>
        
        <div style="background: {color}30; padding: 15px; border-radius: 10px; border: 2px solid {color}; margin: 15px 0;">
            <div style="font-size: 24px; font-weight: bold; color: {color};">{edible_text}</div>
            <div style="font-size: 16px; margin-top: 8px;"><b>{ui['how']}:</b> {how}</div>
            <div style="font-size: 16px; margin-top: 8px; color: #ffeb3b;"><b>{ui['warning']}:</b> {warning}</div>
        </div>
        
        <div style="margin-top: 20px;">
            <h4>{ui['top5']}:</h4>
    """
    
    for cls, prob in top5:
        ci = FUNGI_INFO.get(cls, {"zh": {"name": "?", "edible": True}, "en": {"name": "?", "edible": True}})
        if lang == "中文":
            cn = ci["zh"]["name"]
            cc = "#4CAF50" if ci["zh"]["how"] != "不可食用" else "#F44336"
            label = f"{cls} ({cn})"
        elif lang == "English":
            en = ci["en"]["name"]
            cc = "#4CAF50" if ci["en"]["how"] != "Not edible" else "#F44336"
            label = f"{cls} ({en})"
        else:
            cc = "#4CAF50" if ci["zh"]["how"] != "不可食用" else "#F44336"
            label = f"{cls} ({ci['zh']['name']}/{ci['en']['name']})"
        
        html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{label}</span>
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
        <h3 style="color: #666;">33-class AI Fungi Recognition System</h3>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            lang = gr.Radio(["中文", "English", "Bilingual"], label="Language / 语言", value="中文")
            gr.Markdown("### 📤 Upload Mushroom Photo / 上传蘑菇图片")
            image_input = gr.Image(type="pil", height=400)
            btn = gr.Button("🔍 Recognize / 开始识别", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 Tips / 提示
            - Upload clear mushroom photo / 上传清晰的蘑菇照片
            - Include cap and stem / 包含菌盖和菌柄效果更好
            
            ### ⚠️ Safety / 安全
            **This system is for reference only. Do not rely solely on AI for wild mushroom identification!**
            **本系统仅供参考，请勿仅凭AI判断食用野生蘑菇！**
            """)
        
        with gr.Column():
            gr.Markdown("### 📊 Result / 识别结果")
            image_output = gr.Image(type="pil", height=300)
            result_output = gr.HTML()
    
    btn.click(predict, [image_input, lang], [image_output, result_output])
    
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
        Stanford Summer Program | Fungi Recognition Project<br>
        Functions: Recognition | Toxicity | How to Eat | Warnings<br>
        功能: 识别 | 毒性判断 | 食用方法 | 注意事项
    </div>
    """)

demo.launch(share=True, server_port=7860)
