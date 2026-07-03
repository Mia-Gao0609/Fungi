import os, time, random
from pathlib import Path
from icrawler.builtin import BingImageCrawler

DATA = Path.home() / "fungi_data_100/train"
NUM = 100
MAX_RETRY = 3

FUNGI = {
    "Agaricus_bisporus": "Agaricus bisporus mushroom",
    "Amanita_muscaria": "Amanita muscaria fly agaric mushroom",
    "Amanita_phalloides": "Amanita phalloides death cap mushroom",
    "Amanita_verna": "Amanita verna destroying angel mushroom",
    "Auricularia_auricula": "Auricularia auricula wood ear mushroom",
    "Boletus_edulis": "Boletus edulis mushroom",
    "Calvatia_gigantea": "Calvatia gigantea giant puffball mushroom",
    "Cantharellus_cibarius": "Cantharellus cibarius mushroom",
    "Clitocybe_dealbata": "Clitocybe dealbata sweating fairy mushroom",
    "Coprinus_comatus": "Coprinus comatus shaggy mane mushroom",
    "Cortinarius_rubellus": "Cortinarius rubellus deadly webcap",
    "Dictyophora_duplicata": "Dictyophora duplicata veiled lady mushroom",
    "Entoloma_sinuatum": "Entoloma sinuatum livid entoloma mushroom",
    "Flammulina_velutipes": "Flammulina velutipes enoki mushroom",
    "Galerina_marginata": "Galerina marginata deadly galerina",
    "Grifola_frondosa": "Grifola frondosa maitake mushroom",
    "Gyromitra_esculenta": "Gyromitra esculenta false morel mushroom",
    "Gyromitra_infula": "Gyromitra infula hooded false morel",
    "Hericium_erinaceus": "Hericium erinaceus lion mane mushroom",
    "Laetiporus_sulphureus": "Laetiporus sulphureus chicken of the woods",
    "Lactarius_deliciosus": "Lactarius deliciosus mushroom",
    "Lepiota_brunneoincarnata": "Lepiota brunneoincarnata deadly dapperling",
    "Morchella_esculenta": "Morchella esculenta mushroom",
    "Omphalotus_illudens": "Omphalotus illudens jack o lantern mushroom",
    "Paxillus_involutus": "Paxillus involutus brown roll rim mushroom",
    "Phallus_indusiatus": "Phallus indusiatus bamboo fungus",
    "Pleurotus_ostreatus": "Pleurotus ostreatus mushroom",
    "Russula_subnigricans": "Russula subnigricans blackening russula",
    "Russula_virescens": "Russula virescens green cracking russula",
    "Termitomyces_eurhizus": "Termitomyces eurhizus termite mushroom",
    "Tremella_fuciformis": "Tremella fuciformis white fungus",
    "Tricholoma_matsutake": "Tricholoma matsutake matsutake mushroom",
    "Volvariella_volvacea": "Volvariella volvacea straw mushroom",
}

for idx, (name, query) in enumerate(FUNGI.items(), 1):
    save = str(DATA / name)
    os.makedirs(save, exist_ok=True)
    
    existing = len([f for f in Path(save).glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    if existing >= NUM:
        print(f"[{idx}/33] {name}: 已有 {existing} 张，跳过")
        continue
    
    need = NUM - existing
    print(f"[{idx}/33] {name}: 下载 {need} 张...")
    
    for retry in range(MAX_RETRY):
        try:
            BingImageCrawler(storage={"root_dir": save}).crawl(keyword=query, max_num=need*2)
            break
        except:
            print(f"  重试 {retry+1}/{MAX_RETRY}")
            time.sleep(random.randint(5, 10))
    
    downloaded = len([f for f in Path(save).glob("*") if f.suffix.lower() in ['.jpg','.jpeg','.png']])
    print(f"  ✅ 总计: {downloaded} 张")
    time.sleep(random.randint(3, 6))

print(f"\n🎉 完成！数据在: {DATA}")
