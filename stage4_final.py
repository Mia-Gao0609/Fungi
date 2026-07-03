import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import gradio as gr
from pathlib import Path

MODEL = Path.home() / "fungi_model_best.pth"

# 完整真菌数据
FUNGI_DATA = {
    "Agaricus_bisporus": {
        "zh": {"name": "双孢蘑菇", "how": "炒食、煮汤、烧烤", "warning": "超市常见，放心食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Button Mushroom", "how": "Stir-fry, soup, grill", "warning": "Common in supermarkets, safe to eat", "allergen": "Rare allergies"}
    },
    "Amanita_muscaria": {
        "zh": {"name": "毒蝇伞", "how": "不可食用", "warning": "剧毒！含蝇蕈素，可致幻觉、昏迷、死亡", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Fly Agaric", "how": "Not edible", "warning": "Highly toxic! Contains muscimol, can cause death", "allergen": "Deadly toxic"}
    },
    "Amanita_phalloides": {
        "zh": {"name": "毒鹅膏", "how": "不可食用", "warning": "极毒！致死率极高", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Death Cap", "how": "Not edible", "warning": "Extremely toxic! High fatality rate", "allergen": "Deadly toxic"}
    },
    "Amanita_verna": {
        "zh": {"name": "白毒伞", "how": "不可食用", "warning": "极毒！外观似可食用蘑菇", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Destroying Angel", "how": "Not edible", "warning": "Extremely toxic! Resembles edible mushrooms", "allergen": "Deadly toxic"}
    },
    "Auricularia_auricula": {
        "zh": {"name": "黑木耳", "how": "凉拌、炒食、煮汤", "warning": "需充分泡发后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Wood Ear", "how": "Cold dish, stir-fry, soup", "warning": "Soak thoroughly before cooking", "allergen": "Rare allergies"}
    },
    "Boletus_edulis": {
        "zh": {"name": "美味牛肝菌", "how": "炒食、炖汤、油煎", "warning": "必须彻底煮熟，生吃有毒", "allergen": "部分人可能胃肠不适"},
        "en": {"name": "Porcini", "how": "Stir-fry, soup, pan-fry", "warning": "Must be thoroughly cooked, raw is toxic", "allergen": "May cause stomach upset in some"}
    },
    "Calvatia_gigantea": {
        "zh": {"name": "大马勃", "how": "嫩时炒食、煮汤", "warning": "幼嫩时可食，成熟后不可食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Giant Puffball", "how": "Stir-fry or soup when young", "warning": "Edible when young only", "allergen": "Rare allergies"}
    },
    "Cantharellus_cibarius": {
        "zh": {"name": "鸡油菌", "how": "炒食、炖汤", "warning": "香气浓郁，炒食最佳", "allergen": "少数人可能过敏"},
        "en": {"name": "Chanterelle", "how": "Stir-fry or soup", "warning": "Fragrant, best when stir-fried", "allergen": "Rare allergies"}
    },
    "Clitocybe_dealbata": {
        "zh": {"name": "白霜杯伞", "how": "不可食用", "warning": "有毒！误食可引起中毒", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Sweating Fairy", "how": "Not edible", "warning": "Toxic! Ingestion causes poisoning", "allergen": "Toxic"}
    },
    "Coprinus_comatus": {
        "zh": {"name": "鸡腿菇", "how": "炒食、煮汤", "warning": "需新鲜食用，老化后不可食用", "allergen": "与酒精同食可能不适"},
        "en": {"name": "Shaggy Mane", "how": "Stir-fry or soup", "warning": "Must be fresh, inedible when aged", "allergen": "May react with alcohol"}
    },
    "Cortinarius_rubellus": {
        "zh": {"name": "绯红丝膜菌", "how": "不可食用", "warning": "剧毒！含肾毒素，可致肾衰竭", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Deadly Webcap", "how": "Not edible", "warning": "Highly toxic! Contains nephrotoxins", "allergen": "Deadly toxic"}
    },
    "Dictyophora_duplicata": {
        "zh": {"name": "短裙竹荪", "how": "煮汤、火锅", "warning": "需处理干净后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Veiled Lady", "how": "Soup or hot pot", "warning": "Clean thoroughly before cooking", "allergen": "Rare allergies"}
    },
    "Entoloma_sinuatum": {
        "zh": {"name": "毒粉褶菌", "how": "不可食用", "warning": "有毒！外观与可食用粉褶菌相似", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Livid Entoloma", "how": "Not edible", "warning": "Toxic! Resembles edible species", "allergen": "Toxic"}
    },
    "Flammulina_velutipes": {
        "zh": {"name": "金针菇", "how": "火锅、炒食、煮汤", "warning": "需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Enoki Mushroom", "how": "Hot pot, stir-fry, soup", "warning": "Must be cooked before eating", "allergen": "Rare allergies"}
    },
    "Galerina_marginata": {
        "zh": {"name": "盔孢伞", "how": "不可食用", "warning": "剧毒！含鹅膏毒素，致死率高", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Deadly Galerina", "how": "Not edible", "warning": "Highly toxic! Contains amatoxins", "allergen": "Deadly toxic"}
    },
    "Grifola_frondosa": {
        "zh": {"name": "灰树花", "how": "炒食、炖汤、煮粥", "warning": "又名舞茸，需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Maitake", "how": "Stir-fry, soup, porridge", "warning": "Also known as Maitake, must be cooked", "allergen": "Rare allergies"}
    },
    "Gyromitra_esculenta": {
        "zh": {"name": "鹿花菌", "how": "不可食用", "warning": "有毒！含鹿花菌素，可致中毒", "allergen": "剧毒，禁止接触"},
        "en": {"name": "False Morel", "how": "Not edible", "warning": "Toxic! Contains gyromitrin", "allergen": "Deadly toxic"}
    },
    "Gyromitra_infula": {
        "zh": {"name": "褐鹿花菌", "how": "不可食用", "warning": "有毒！与鹿花菌同属", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Hooded False Morel", "how": "Not edible", "warning": "Toxic! Related to False Morel", "allergen": "Deadly toxic"}
    },
    "Hericium_erinaceus": {
        "zh": {"name": "猴头菇", "how": "炖汤、煮粥、炒食", "warning": "养胃佳品，需充分煮熟", "allergen": "少数人可能过敏"},
        "en": {"name": "Lion's Mane", "how": "Soup, porridge, stir-fry", "warning": "Good for stomach health, must be cooked", "allergen": "Rare allergies"}
    },
    "Laetiporus_sulphureus": {
        "zh": {"name": "硫色绚孔菌", "how": "炒食、炖汤", "warning": "鸡肉味，需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Chicken of the Woods", "how": "Stir-fry or soup", "warning": "Tastes like chicken, must be cooked", "allergen": "Rare allergies"}
    },
    "Lactarius_deliciosus": {
        "zh": {"name": "松乳菇", "how": "炒食、炖汤", "warning": "需煮熟后食用，有松香味", "allergen": "少数人可能过敏"},
        "en": {"name": "Saffron Milk Cap", "how": "Stir-fry or soup", "warning": "Must be cooked, has pine aroma", "allergen": "Rare allergies"}
    },
    "Lepiota_brunneoincarnata": {
        "zh": {"name": "褐鳞小伞", "how": "不可食用", "warning": "剧毒！含鹅膏毒素，致死率高", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Deadly Dapperling", "how": "Not edible", "warning": "Highly toxic! Contains amatoxins", "allergen": "Deadly toxic"}
    },
    "Morchella_esculenta": {
        "zh": {"name": "羊肚菌", "how": "炖汤、炒食", "warning": "必须彻底煮熟，不可生吃", "allergen": "少数人可能过敏"},
        "en": {"name": "Morel", "how": "Soup or stir-fry", "warning": "Must be thoroughly cooked, never raw", "allergen": "Rare allergies"}
    },
    "Omphalotus_illudens": {
        "zh": {"name": "发光脐菇", "how": "不可食用", "warning": "有毒！夜间发光，误食可引起中毒", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Jack-o'-Lantern", "how": "Not edible", "warning": "Toxic! Bioluminescent at night", "allergen": "Toxic"}
    },
    "Paxillus_involutus": {
        "zh": {"name": "卷边桩菇", "how": "不可食用", "warning": "有毒！可引起溶血反应", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Brown Roll-Rim", "how": "Not edible", "warning": "Toxic! Can cause hemolysis", "allergen": "Toxic"}
    },
    "Phallus_indusiatus": {
        "zh": {"name": "竹荪", "how": "炖汤、火锅", "warning": "名贵菌类，需处理干净后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Bamboo Fungus", "how": "Soup or hot pot", "warning": "Premium fungus, clean thoroughly", "allergen": "Rare allergies"}
    },
    "Pleurotus_ostreatus": {
        "zh": {"name": "平菇", "how": "炒食、煮汤、火锅", "warning": "常见食用菌，需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Oyster Mushroom", "how": "Stir-fry, soup, hot pot", "warning": "Common edible mushroom, must be cooked", "allergen": "Rare allergies"}
    },
    "Russula_subnigricans": {
        "zh": {"name": "亚稀褶黑菇", "how": "不可食用", "warning": "有毒！外观与可食用红菇相似", "allergen": "剧毒，禁止接触"},
        "en": {"name": "Blackening Russula", "how": "Not edible", "warning": "Toxic! Resembles edible Russula", "allergen": "Toxic"}
    },
    "Russula_virescens": {
        "zh": {"name": "变绿红菇", "how": "炒食", "warning": "需确认品种，部分红菇有毒", "allergen": "少数人可能过敏"},
        "en": {"name": "Green Cracking Russula", "how": "Stir-fry", "warning": "Verify species, some are toxic", "allergen": "Rare allergies"}
    },
    "Termitomyces_eurhizus": {
        "zh": {"name": "鸡枞菌", "how": "炒食、炖汤、油炸", "warning": "白蚁共生菌，需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Termite Mushroom", "how": "Stir-fry, soup, deep-fry", "warning": "Termite symbiotic fungus, must be cooked", "allergen": "Rare allergies"}
    },
    "Tremella_fuciformis": {
        "zh": {"name": "银耳", "how": "炖汤、煮粥、凉拌", "warning": "滋补佳品，需充分泡发后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "White Fungus", "how": "Soup, porridge, cold dish", "warning": "Nutritious tonic, soak thoroughly", "allergen": "Rare allergies"}
    },
    "Tricholoma_matsutake": {
        "zh": {"name": "松茸", "how": "烤食、炖汤、炒食", "warning": "名贵食用菌，需轻处理保留香气", "allergen": "少数人可能过敏"},
        "en": {"name": "Matsutake", "how": "Grill, soup, stir-fry", "warning": "Premium fungus, handle gently", "allergen": "Rare allergies"}
    },
    "Volvariella_volvacea": {
        "zh": {"name": "草菇", "how": "炒食、煮汤", "warning": "高温菇类，需煮熟后食用", "allergen": "少数人可能过敏"},
        "en": {"name": "Straw Mushroom", "how": "Stir-fry or soup", "warning": "Heat-loving fungus, must be cooked", "allergen": "Rare allergies"}
    },
}

# 加载模型
checkpoint = torch.load(MODEL, map_location="cpu")
CLASS_NAMES = checkpoint['classes']

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ========== 功能1：图片识别 ==========
def predict_image(img, lang):
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
    top3 = [(CLASS_NAMES[i], probs[0][j].item()) for j, i in enumerate(probs.topk(3)[1][0])]
    
    return format_result(pred_class, confidence, top3, lang, is_image=True)

# ========== 功能2：种类查询 ==========
def query_fungi(fungi_name, lang):
    if not fungi_name:
        return "Please select a fungi / 请选择一种真菌"
    
    data = FUNGI_DATA.get(fungi_name, {
        "zh": {"name": "未知", "how": "未知", "warning": "暂无信息", "allergen": "未知"},
        "en": {"name": "Unknown", "how": "Unknown", "warning": "No information", "allergen": "Unknown"}
    })
    
    is_edible = data["zh"]["how"] != "不可食用"
    
    if lang == "中文":
        name = data["zh"]["name"]
        how = data["zh"]["how"]
        warning = data["zh"]["warning"]
        allergen = data["zh"]["allergen"]
        edible_text = "✅ 可食用" if is_edible else "❌ 有毒"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 真菌信息查询"
        how_label = "食用方法"
        warn_label = "注意事项"
        allergen_label = "过敏源/特殊提示"
    elif lang == "English":
        name = data["en"]["name"]
        how = data["en"]["how"]
        warning = data["en"]["warning"]
        allergen = data["en"]["allergen"]
        edible_text = "✅ Edible" if is_edible else "❌ Poisonous"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 Fungi Information"
        how_label = "How to Eat"
        warn_label = "Warning"
        allergen_label = "Allergen/Special Notes"
    else:
        name = f"{data['zh']['name']} / {data['en']['name']}"
        how = f"{data['zh']['how']} / {data['en']['how']}"
        warning = f"{data['zh']['warning']} / {data['en']['warning']}"
        allergen = f"{data['zh']['allergen']} / {data['en']['allergen']}"
        edible_text = "✅ 可食用 / Edible" if is_edible else "❌ 有毒 / Poisonous"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 真菌信息查询 / Fungi Information"
        how_label = "食用方法 / How to Eat"
        warn_label = "注意事项 / Warning"
        allergen_label = "过敏源/特殊提示 / Allergen/Special Notes"
    
    return f"""
    <div style="padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
        <h2>{title}</h2>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; color: #333; margin: 15px 0;">
            <div style="font-size: 26px; font-weight: bold; color: {color};">{fungi_name}</div>
            <div style="font-size: 20px; color: #666; margin-top: 5px;">{name}</div>
        </div>
        <div style="background: {color}25; padding: 15px; border-radius: 10px; border: 2px solid {color}; margin: 15px 0;">
            <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 10px;">{edible_text}</div>
            <div style="font-size: 15px; margin-bottom: 8px;"><b>{how_label}:</b> {how}</div>
            <div style="font-size: 15px; margin-bottom: 8px; color: #ffeb3b; font-weight: bold;"><b>⚠️ {warn_label}:</b> {warning}</div>
            <div style="font-size: 15px; color: #ffc107;"><b>🔔 {allergen_label}:</b> {allergen}</div>
        </div>
    </div>
    """

# ========== 格式化识别结果 ==========
def format_result(pred_class, confidence, top3, lang, is_image=False):
    data = FUNGI_DATA.get(pred_class, {
        "zh": {"name": "未知", "how": "未知", "warning": "暂无信息", "allergen": "未知"},
        "en": {"name": "Unknown", "how": "Unknown", "warning": "No information", "allergen": "Unknown"}
    })
    
    is_edible = data["zh"]["how"] != "不可食用"
    
    if lang == "中文":
        name = data["zh"]["name"]
        how = data["zh"]["how"]
        warning = data["zh"]["warning"]
        allergen = data["zh"]["allergen"]
        edible_text = "✅ 可食用" if is_edible else "❌ 有毒"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 识别结果"
        conf_label = "置信度"
        how_label = "食用方法"
        warn_label = "注意事项"
        allergen_label = "过敏源/特殊提示"
        top_label = "Top 3 候选"
        low_conf = "⚠️ 置信度低，结果可能不准确"
    elif lang == "English":
        name = data["en"]["name"]
        how = data["en"]["how"]
        warning = data["en"]["warning"]
        allergen = data["en"]["allergen"]
        edible_text = "✅ Edible" if is_edible else "❌ Poisonous"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 Recognition Result"
        conf_label = "Confidence"
        how_label = "How to Eat"
        warn_label = "Warning"
        allergen_label = "Allergen/Special Notes"
        top_label = "Top 3 Candidates"
        low_conf = "⚠️ Low confidence, result may be inaccurate"
    else:
        name = f"{data['zh']['name']} / {data['en']['name']}"
        how = f"{data['zh']['how']} / {data['en']['how']}"
        warning = f"{data['zh']['warning']} / {data['en']['warning']}"
        allergen = f"{data['zh']['allergen']} / {data['en']['allergen']}"
        edible_text = "✅ 可食用 / Edible" if is_edible else "❌ 有毒 / Poisonous"
        color = "#4CAF50" if is_edible else "#F44336"
        title = "🍄 识别结果 / Recognition Result"
        conf_label = "置信度 / Confidence"
        how_label = "食用方法 / How to Eat"
        warn_label = "注意事项 / Warning"
        allergen_label = "过敏源/特殊提示 / Allergen/Special Notes"
        top_label = "Top 3 候选 / Top 3 Candidates"
        low_conf = "⚠️ 置信度低 / Low confidence"
    
    low_conf_html = ""
    if confidence < 0.3:
        low_conf_html = f"<div style='background: #ff9800; color: white; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; text-align: center;'>{low_conf}</div>"
    
    html = f"""
    <div style="padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
        {low_conf_html}
        <h2>{title}</h2>
        <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 10px; color: #333; margin: 15px 0;">
            <div style="font-size: 26px; font-weight: bold; color: {color};">{pred_class}</div>
            <div style="font-size: 20px; color: #666; margin-top: 5px;">{name}</div>
            <div style="margin-top: 10px; font-size: 16px;">{conf_label}: <b style="color: {color}; font-size: 20px;">{confidence*100:.1f}%</b></div>
        </div>
        <div style="background: {color}25; padding: 15px; border-radius: 10px; border: 2px solid {color}; margin: 15px 0;">
            <div style="font-size: 22px; font-weight: bold; color: {color}; margin-bottom: 10px;">{edible_text}</div>
            <div style="font-size: 15px; margin-bottom: 8px;"><b>{how_label}:</b> {how}</div>
            <div style="font-size: 15px; margin-bottom: 8px; color: #ffeb3b; font-weight: bold;"><b>⚠️ {warn_label}:</b> {warning}</div>
            <div style="font-size: 15px; color: #ffc107;"><b>🔔 {allergen_label}:</b> {allergen}</div>
        </div>
        <div>
            <h4>{top_label}</h4>
    """
    
    for cls, prob in top3:
        cd = FUNGI_DATA.get(cls, {"zh": {"name": "?", "how": "?"}, "en": {"name": "?", "how": "Not edible"}})
        cls_edible = cd["zh"]["how"] != "不可食用"
        cc = "#4CAF50" if cls_edible else "#F44336"
        
        if lang == "中文":
            label = f"{cls} ({cd['zh']['name']})"
        elif lang == "English":
            label = f"{cls} ({cd['en']['name']})"
        else:
            label = f"{cls} ({cd['zh']['name']}/{cd['en']['name']})"
        
        html += f"""
            <div style="margin: 8px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 14px;">{label}</span>
                    <span style="color: {cc}; font-weight: bold; font-size: 14px;">{prob*100:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.25); border-radius: 8px; height: 20px;">
                    <div style="background: {cc}; width: {int(prob*100)}%; height: 100%; border-radius: 8px;"></div>
                </div>
            </div>
        """
    
    html += "</div></div>"
    return None, html

# ========== 创建界面 ==========
with gr.Blocks(title="🍄 Fungi Classifier 33", theme=gr.themes.Soft()) as demo:
    # 顶部警告
    gr.Markdown("""
    <div style="background: linear-gradient(90deg, #ff4444, #ff6600); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; font-weight: bold; font-size: 16px;">
        ⚠️ 警告 / WARNING ⚠️<br>
        本系统准确率仅约58%，不可用于实际食用判断！<br>
        This system is only ~58% accurate. DO NOT use for actual edibility determination!<br>
        <span style="color: #ffeb3b;">颜色鲜艳的蘑菇通常有毒！/ Brightly colored mushrooms are usually poisonous!</span>
    </div>
    """)
    
    gr.Markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 42px; margin-bottom: 5px;">🍄 Fungi Classifier</h1>
        <h3 style="color: #666; margin-top: 0;">33-class AI Fungi Recognition System</h3>
    </div>
    """)
    
    # 全局语言选择
    with gr.Row():
        lang_select = gr.Radio(
            choices=["中文", "English", "Bilingual"],
            value="中文",
            label="Language / 语言"
        )
    
    # 标签页
    with gr.Tab("📷 Image Recognition / 图片识别"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Upload / 上传")
                image_input = gr.Image(type="pil", height=380)
                btn_recognize = gr.Button("🔍 Recognize / 开始识别", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                ### 💡 Tips / 提示
                - Upload clear mushroom photo / 上传清晰的蘑菇照片
                - Include cap and stem / 包含菌盖和菌柄效果更好
                """)
            
            with gr.Column():
                gr.Markdown("### 📊 Result / 结果")
                image_output = gr.Image(type="pil", height=280)
                result_output = gr.HTML()
        
        btn_recognize.click(
            fn=predict_image,
            inputs=[image_input, lang_select],
            outputs=[image_output, result_output]
        )
    
    with gr.Tab("📚 Fungi Database / 真菌数据库"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 Query / 查询")
                fungi_select = gr.Dropdown(
                    choices=CLASS_NAMES,
                    value=CLASS_NAMES[0],
                    label="Select Fungi / 选择真菌"
                )
                btn_query = gr.Button("🔍 Query / 查询", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                ### 💡 Info / 说明
                直接查询33种真菌的详细信息，无需上传图片。<br>
                Query detailed info of 33 fungi types without uploading images.
                """)
            
            with gr.Column():
                gr.Markdown("### 📊 Information / 信息")
                query_output = gr.HTML()
        
        btn_query.click(
            fn=query_fungi,
            inputs=[fungi_select, lang_select],
            outputs=query_output
        )
    
    gr.Markdown("""
    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px; margin-top: 20px;">
        Stanford Summer Program | Fungi Recognition Project<br>
        Accuracy: ~58% | For demonstration purposes only
    </div>
    """)

demo.launch(share=True, server_port=7860)
