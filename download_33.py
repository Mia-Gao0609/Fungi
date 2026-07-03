import os, time
from pathlib import Path
from icrawler.builtin import BingImageCrawler

DATA = Path.home() / "fungi_data_33/train"
NUM = 30

FUNGI = {
    # 可食用
    "Agaricus_bisporus": "Agaricus bisporus mushroom",
    "Boletus_edulis": "Boletus edulis mushroom",
    "Cantharellus_cibarius": "Cantharellus cibarius mushroom",
    "Lactarius_deliciosus": "Lactarius deliciosus mushroom",
    "Morchella_esculenta": "Morchella esculenta mushroom",
    "Pleurotus_ostreatus": "Pleurotus ostreatus mushroom",
    "Hericium_erinaceus": "Hericium erinaceus lion mane mushroom",
    "Grifola_frondosa": "Grifola frondosa maitake mushroom",
    "Volvariella_volvacea": "Volvariella volvacea straw mushroom",
    "Flammulina_velutipes": "Flammulina velutipes enoki mushroom",
    "Tremella_fuciformis": "Tremella fuciformis white fungus",
    "Auricularia_auricula": "Auricularia auricula wood ear mushroom",
    "Coprinus_comatus": "Coprinus comatus shaggy mane mushroom",
    "Calvatia_gigantea": "Calvatia gigantea giant puffball mushroom",
    "Laetiporus_sulphureus": "Laetiporus sulphureus chicken of the woods",
    "Russula_virescens": "Russula virescens green cracking russula",
    "Tricholoma_matsutake": "Tricholoma matsutake matsutake mushroom",
    "Termitomyces_eurhizus": "Termitomyces eurhizus termite mushroom",
    "Phallus_indusiatus": "Phallus indusiatus bamboo fungus",
    "Dictyophora_duplicata": "Dictyophora duplicata veiled lady mushroom",
    # 有毒
    "Amanita_muscaria": "Amanita muscaria fly agaric mushroom",
    "Amanita_phalloides": "Amanita phalloides death cap mushroom",
    "Amanita_verna": "Amanita verna destroying angel mushroom",
    "Cortinarius_rubellus": "Cortinarius rubellus deadly webcap",
    "Gyromitra_esculenta": "Gyromitra esculenta false morel mushroom",
    "Gyromitra_infula": "Gyromitra infula hooded false morel",
    "Galerina_marginata": "Galerina marginata deadly galerina",
    "Lepiota_brunneoincarnata": "Lepiota brunneoincarnata deadly dapperling",
    "Entoloma_sinuatum": "Entoloma sinuatum livid entoloma mushroom",
    "Clitocybe_dealbata": "Clitocybe dealbata sweating fairy mushroom",
    "Omphalotus_illudens": "Omphalotus illudens jack o lantern mushroom",
    "Russula_subnigricans": "Russula subnigricans blackening russula",
    "Paxillus_involutus": "Paxillus involutus brown roll rim mushroom",
}

for idx, (name, query) in enumerate(FUNGI.items(), 1):
    save = str(DATA / name)
    os.makedirs(save, exist_ok=True)
    print(f"[{idx}/33] {name}")
    try:
        BingImageCrawler(storage={"root_dir": save}).crawl(keyword=query, max_num=NUM)
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ❌ {str(e)[:50]}")
    time.sleep(2)

print(f"\nData saved to: {DATA}")
