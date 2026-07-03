from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# 33种真菌
FUNGI_CLASSES = [
    "jianshouqing", "boletus_edulis", "cantharellus_cibarius", 
    "morchella_esculenta", "agaricus_bisporus", "pleurotus_ostreatus",
    "lentinula_edodes", "volvariella_volvacea", "hericium_erinaceus",
    "tremella_fuciformis", "auricularia_auricula", "coprinus_comatus",
    "armillaria_mellea", "suillus_brevipes", "tricholoma_matsutake",
    "amanita_phalloides", "amanita_muscaria", "gyromitra_esculenta",
    "russula_subnigricans", "amanita_verna", "amanita_virosa",
    "cortinarius_rubellus", "cortinarius_orellanus", "galerina_marginata",
    "lepiota_brunneoincarnata", "entoloma_sinuatum", "clitocybe_dealbata",
    "omphalotus_illudens", "paxillus_involutus", "podostroma_cornu_damae",
    "gyromitra_infula", "verpa_bohemica", "morchella_semilibera"
]

# 加载预训练模型
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(FUNGI_CLASSES),
    ignore_mismatched_sizes=True
)

print(f"Model loaded with {len(FUNGI_CLASSES)} classes")
print("Classes:", FUNGI_CLASSES)

# 保存
model.save_pretrained("./fungi_model")
print("Saved to ./fungi_model")
