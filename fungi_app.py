import gradio as gr
from dataclasses import dataclass
from typing import List
from enum import Enum

class Edibility(Enum):
    EDIBLE = ("可食用", "Edible")
    CONDITIONAL = ("需处理", "Requires Preparation")
    TOXIC = ("有毒", "Toxic")
    DEADLY = ("剧毒", "Deadly")
    UNKNOWN = ("未知", "Unknown")

@dataclass
class FungiInfo:
    latin_name: str
    chinese_name: str
    english_name: str
    edibility: Edibility
    preparation_cn: str
    preparation_en: str
    warnings_cn: List[str]
    warnings_en: List[str]
    symptoms_cn: str = ""
    symptoms_en: str = ""

FUNGI_DATABASE = {
    "jianshouqing": FungiInfo("Boletus bicolor", "见手青", "Two-colored Bolete", Edibility.CONDITIONAL, "彻底煮熟，切片后高温油炒或油炸", "Must be thoroughly cooked. Slice and stir-fry or deep-fry.", ["生食有毒", "与某些有毒牛肝菌外观相似"], ["Raw is toxic", "Similar to poisonous boletes"], "幻觉、肠胃不适（未充分加热时）", "Hallucinations, gastrointestinal discomfort (if undercooked)"),
    "boletus_edulis": FungiInfo("Boletus edulis", "美味牛肝菌", "Porcini", Edibility.EDIBLE, "切片晒干或鲜食，需彻底煮熟", "Slice and dry or eat fresh. Must be thoroughly cooked.", ["部分人群可能过敏"], ["Some people may be allergic"]),
    "cantharellus_cibarius": FungiInfo("Cantharellus cibarius", "鸡油菌", "Chanterelle", Edibility.EDIBLE, "洗净去沙，可炒、炖、做汤", "Wash thoroughly, can be stir-fried, stewed, or in soup.", ["需彻底清洗去除泥沙"], ["Must be thoroughly cleaned to remove sand"]),
    "morchella_esculenta": FungiInfo("Morchella esculenta", "羊肚菌", "Morel", Edibility.CONDITIONAL, "必须彻底煮熟！生食有毒", "Must be thoroughly cooked! Raw is toxic.", ["生食有毒", "与鹿花菌极相似"], ["Raw is toxic", "Extremely similar to False Morel"]),
    "agaricus_bisporus": FungiInfo("Agaricus bisporus", "双孢蘑菇", "White Button", Edibility.EDIBLE, "洗净后可炒、煮、烤、做汤", "Wash and cook by stir-frying, boiling, roasting, or soup.", ["需彻底煮熟"], ["Must be thoroughly cooked"]),
    "pleurotus_ostreatus": FungiInfo("Pleurotus ostreatus", "平菇", "Oyster Mushroom", Edibility.EDIBLE, "洗净去蒂，可炒、涮火锅、油炸", "Wash and remove stem, can be stir-fried, hot-pot, or deep-fried.", ["痛风患者少食"], ["People with gout should eat sparingly"]),
    "lentinula_edodes": FungiInfo("Lentinula edodes", "香菇", "Shiitake", Edibility.EDIBLE, "泡发后彻底煮熟", "Soak and cook thoroughly.", ["可能过敏"], ["May cause allergic reactions"]),
    "volvariella_volvacea": FungiInfo("Volvariella volvacea", "草菇", "Straw Mushroom", Edibility.EDIBLE, "洗净后可炒、煮、煲汤", "Wash and cook by stir-frying, boiling, or soup.", ["无特殊"], ["None"]),
    "hericium_erinaceus": FungiInfo("Hericium erinaceus", "猴头菇", "Lion's Mane", Edibility.EDIBLE, "泡发后炖、煮、蒸", "Soak and stew, boil, or steam.", ["无特殊"], ["None"]),
    "tremella_fuciformis": FungiInfo("Tremella fuciformis", "银耳", "Snow Fungus", Edibility.EDIBLE, "泡发后炖、煮甜品", "Soak and stew, or cook in sweet soup.", ["无特殊"], ["None"]),
    "auricularia_auricula": FungiInfo("Auricularia auricula", "黑木耳", "Wood Ear", Edibility.EDIBLE, "泡发后炒、煮、凉拌", "Soak and stir-fry, boil, or cold dish.", ["无特殊"], ["None"]),
    "coprinus_comatus": FungiInfo("Coprinus comatus", "鸡腿菇", "Shaggy Mane", Edibility.EDIBLE, "彻底煮熟，不宜与酒同食", "Cook thoroughly, avoid alcohol.", ["与酒同食可能不适"], ["May cause discomfort with alcohol"]),
    "armillaria_mellea": FungiInfo("Armillaria mellea", "蜜环菌", "Honey Fungus", Edibility.EDIBLE, "彻底煮熟", "Cook thoroughly.", ["部分有毒变种"], ["Some toxic variants exist"]),
    "suillus_brevipes": FungiInfo("Suillus brevipes", "短柄粘盖牛肝菌", "Short-stalked Suillus", Edibility.EDIBLE, "去皮后煮熟", "Peel and cook thoroughly.", ["必须去皮"], ["Must peel before cooking"]),
    "tricholoma_matsutake": FungiInfo("Tricholoma matsutake", "松茸", "Matsutake", Edibility.EDIBLE, "切片煎、烤、炖", "Slice and pan-fry, roast, or stew.", ["珍贵稀有"], ["Rare and precious"]),
    "amanita_phalloides": FungiInfo("Amanita phalloides", "毒鹅膏", "Death Cap", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["致死率极高", "50g即可致命"], ["Extremely high fatality rate", "50g can be fatal"], "潜伏期6-24h，呕吐腹泻→假愈期→肝肾功能衰竭", "Latency 6-24h, vomiting/diarrhea → false recovery → liver/kidney failure"),
    "amanita_muscaria": FungiInfo("Amanita muscaria", "毒蝇伞", "Fly Agaric", Edibility.TOXIC, "有毒！不可食用！", "TOXIC! Do not consume!", ["致幻、精神错乱", "儿童易被鲜艳颜色吸引"], ["Causes hallucinations and delirium", "Bright colors attract children"], "恶心、头晕、幻觉、谵妄、抽搐", "Nausea, dizziness, hallucinations, delirium, seizures"),
    "gyromitra_esculenta": FungiInfo("Gyromitra esculenta", "鹿花菌", "False Morel", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["与羊肚菌极相似，极易误采", "死亡率极高"], ["Extremely similar to true morels", "Extremely high mortality rate"], "呕吐、腹泻、头晕、黄疸、肝衰竭", "Vomiting, diarrhea, dizziness, jaundice, liver failure"),
    "russula_subnigricans": FungiInfo("Russula subnigricans", "亚稀褶黑菇", "Blackening Brittlegill", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["与可食红菇极相似", "是中国南方中毒死亡主要原因"], ["Extremely similar to edible Russula", "Leading cause of poisoning deaths in southern China"], "恶心、呕吐、肌肉酸痛、酱油色尿、急性肾衰竭", "Nausea, vomiting, muscle pain, dark urine, acute kidney failure"),
    "amanita_verna": FungiInfo("Amanita verna", "白毒伞", "Destroying Angel", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["极毒", "与可食蘑菇极相似"], ["Extremely toxic", "Very similar to edible mushrooms"], "呕吐、腹泻、痉挛、肝衰竭、死亡", "Vomiting, diarrhea, cramps, liver failure, death"),
    "amanita_virosa": FungiInfo("Amanita virosa", "鳞柄白毒伞", "European Destroying Angel", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["极毒", "与可食蘑菇极相似"], ["Extremely toxic", "Very similar to edible mushrooms"], "呕吐、腹泻、痉挛、肝衰竭、死亡", "Vomiting, diarrhea, cramps, liver failure, death"),
    "cortinarius_rubellus": FungiInfo("Cortinarius rubellus", "红丝膜菌", "Deadly Webcap", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["肾衰竭", "潜伏期长"], ["Kidney failure", "Long latency period"], "类似流感症状→肾衰竭", "Flu-like symptoms → kidney failure"),
    "cortinarius_orellanus": FungiInfo("Cortinarius orellanus", "奥来丝膜菌", "Fool's Webcap", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["肾衰竭", "极毒"], ["Kidney failure", "Extremely toxic"], "类似流感症状→肾衰竭", "Flu-like symptoms → kidney failure"),
    "galerina_marginata": FungiInfo("Galerina marginata", "纹缘盔孢伞", "Deadly Galerina", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["极毒", "与可食金针菇极相似"], ["Extremely toxic", "Very similar to edible enoki"], "呕吐、腹泻、肝衰竭、死亡", "Vomiting, diarrhea, liver failure, death"),
    "lepiota_brunneoincarnata": FungiInfo("Lepiota brunneoincarnata", "褐鳞小伞", "Deadly Dapperling", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["极毒", "小型但致命"], ["Extremely toxic", "Small but deadly"], "呕吐、腹泻、肝衰竭、死亡", "Vomiting, diarrhea, liver failure, death"),
    "entoloma_sinuatum": FungiInfo("Entoloma sinuatum", "毒粉褶菌", "Livid Entoloma", Edibility.TOXIC, "有毒！不可食用！", "TOXIC! Do not consume!", ["肠胃炎", "与可食粉褶菌相似"], ["Gastroenteritis", "Similar to edible Entoloma"], "恶心、呕吐、腹泻、腹痛", "Nausea, vomiting, diarrhea, abdominal pain"),
    "clitocybe_dealbata": FungiInfo("Clitocybe dealbata", "白霜杯伞", "Ivory Funnel", Edibility.TOXIC, "有毒！不可食用！", "TOXIC! Do not consume!", ["出汗中毒", "与可食杯伞相似"], ["Sweating syndrome", "Similar to edible Clitocybe"], "大量出汗、流涎、流泪、腹痛", "Profuse sweating, salivation, tearing, abdominal pain"),
    "omphalotus_illudens": FungiInfo("Omphalotus illudens", "杰克灯笼菌", "Jack O'Lantern", Edibility.TOXIC, "有毒！不可食用！", "TOXIC! Do not consume!", ["似鸡油菌", "发光"], ["Looks like chanterelle", "Bioluminescent"], "恶心、呕吐、腹泻、痉挛", "Nausea, vomiting, diarrhea, cramps"),
    "paxillus_involutus": FungiInfo("Paxillus involutus", "卷缘桩菇", "Brown Roll-Rim", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["免疫溶血", "反复食用毒性累积"], ["Immune hemolysis", "Toxicity accumulates with repeated consumption"], "呕吐、腹泻、溶血、肾衰竭", "Vomiting, diarrhea, hemolysis, kidney failure"),
    "podostroma_cornu_damae": FungiInfo("Podostroma cornu-damae", "鹿角炭角菌", "Poison Fire Coral", Edibility.DEADLY, "剧毒！不可食用！", "DEADLY POISONOUS! Do not consume!", ["多器官衰竭", "无特效解毒剂"], ["Multi-organ failure", "No specific antidote"], "皮肤接触也可能中毒", "Skin contact may also cause poisoning"),
    "gyromitra_infula": FungiInfo("Gyromitra infula", "赭鹿花菌", "Hooded False Morel", Edibility.CONDITIONAL, "充分煮熟，但建议避免食用", "Must be thoroughly cooked, but avoid if possible.", ["处理不当有毒", "与鹿花菌同族"], ["Toxic if undercooked", "Same family as False Morel"], "类似鹿花菌中毒", "Similar to False Morel poisoning"),
    "verpa_bohemica": FungiInfo("Verpa bohemica", "波氏钟菌", "Early Morel", Edibility.CONDITIONAL, "充分煮熟，但建议避免食用", "Must be thoroughly cooked, but avoid if possible.", ["似羊肚菌", "处理不当有毒"], ["Looks like morel", "Toxic if undercooked"], "类似羊肚菌/鹿花菌中毒", "Similar to morel/False Morel poisoning"),
    "morchella_semilibera": FungiInfo("Morchella semilibera", "半开羊肚菌", "Half-free Morel", Edibility.CONDITIONAL, "必须彻底煮熟", "Must be thoroughly cooked.", ["处理不当有毒", "与羊肚菌相似"], ["Toxic if undercooked", "Similar to true morel"], "类似羊肚菌中毒", "Similar to morel poisoning"),
}

def generate_report(fungi_key, language):
    fungi = FUNGI_DATABASE.get(fungi_key, FUNGI_DATABASE["jianshouqing"])
    edibility_cn, edibility_en = fungi.edibility.value
    
    emoji = {
        Edibility.EDIBLE: "✅",
        Edibility.CONDITIONAL: "⚠️",
        Edibility.TOXIC: "❌",
        Edibility.DEADLY: "☠️",
        Edibility.UNKNOWN: "❓"
    }.get(fungi.edibility, "❓")
    
    if language == "中文":
        report = f"""
{'='*60}
🍄 真菌识别报告
{'='*60}

名称：{fungi.chinese_name}
拉丁学名：{fungi.latin_name}

可食用性：{emoji} {edibility_cn}

处理方法：
{fungi.preparation_cn}

警告：
"""
        for w in fungi.warnings_cn:
            report += f"  • {w}\n"
        
        if fungi.symptoms_cn:
            report += f"\n中毒症状：\n{fungi.symptoms_cn}\n"
        
        report += f"\n{'='*60}\n⚠️ 请勿仅凭AI识别采食野生蘑菇！\n⚠️ 中毒后立即就医，保留样本！\n{'='*60}"
        
    elif language == "English":
        report = f"""
{'='*60}
🍄 Fungi Identification Report
{'='*60}

Name: {fungi.english_name}
Latin: {fungi.latin_name}

Edibility: {emoji} {edibility_en}

Preparation:
{fungi.preparation_en}

Warnings:
"""
        for w in fungi.warnings_en:
            report += f"  • {w}\n"
        
        if fungi.symptoms_en:
            report += f"\nSymptoms:\n{fungi.symptoms_en}\n"
        
        report += f"\n{'='*60}\n⚠️ DO NOT consume wild mushrooms based solely on AI!\n⚠️ Seek medical attention immediately if poisoned!\n{'='*60}"
        
    else:  # 双语
        report = f"""
{'='*60}
🍄 真菌识别报告 / Fungi Identification Report
{'='*60}

名称 / Name: {fungi.chinese_name} / {fungi.english_name}
拉丁 / Latin: {fungi.latin_name}

可食用性 / Edibility: {emoji} {edibility_cn} / {edibility_en}

处理方法 / Preparation:
  CN: {fungi.preparation_cn}
  EN: {fungi.preparation_en}

警告 / Warnings:
"""
        for w_cn, w_en in zip(fungi.warnings_cn, fungi.warnings_en):
            report += f"  • {w_cn} / {w_en}\n"
        
        if fungi.symptoms_cn:
            report += f"\n中毒症状 / Symptoms:\n  CN: {fungi.symptoms_cn}\n  EN: {fungi.symptoms_en}\n"
        
        report += f"\n{'='*60}\n⚠️ 请勿仅凭AI识别采食野生蘑菇！\n⚠️ DO NOT consume wild mushrooms based solely on AI!\n⚠️ 中毒后立即就医，保留样本！\n⚠️ Seek medical attention immediately if poisoned!\n{'='*60}"
    
    return report

with gr.Blocks(title="Fungi Identifier / 真菌识别器") as demo:
    gr.Markdown("# 🍄 真菌识别器 / Fungi Identifier")
    gr.Markdown(f"支持 {len(FUNGI_DATABASE)} 种真菌 / Supports {len(FUNGI_DATABASE)} fungi species")
    
    with gr.Row():
        with gr.Column():
            # 图片上传框
            input_image = gr.Image(label="上传真菌图片 / Upload Fungi Image")
            
            fungi_select = gr.Dropdown(
                choices=[(f"{v.chinese_name} / {v.english_name}", k) for k, v in FUNGI_DATABASE.items()],
                value="jianshouqing",
                label="选择真菌 / Select Fungi"
            )
            language_select = gr.Dropdown(
                choices=["中文", "English", "双语 / Bilingual"],
                value="双语 / Bilingual",
                label="选择语言 / Select Language"
            )
            btn = gr.Button("识别 / Identify", variant="primary")
        
        with gr.Column():
            # 显示原图
            output_image = gr.Image(label="上传图片 / Uploaded Image")
            output_text = gr.Textbox(label="识别报告 / Report", lines=25)
    
    # 点击识别：显示原图 + 报告
    btn.click(fn=lambda img, key, lang: (img, generate_report(key, lang)), 
              inputs=[input_image, fungi_select, language_select], 
              outputs=[output_image, output_text])

demo.launch(server_name="0.0.0.0", server_port=7860)
