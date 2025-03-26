import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa

print(f" [i] Start analyzing, total 8 steps...")
print(f" [i] Setting up...")

# 強制 Qt 在 Linux 環境下使用 X11
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 防止 NumPy 佔用過多記憶體
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 固定隨機種子確保可重現性
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

def load_metadata_labels(metadata_path, df_index, target_tissues=None):
    print(f" [i] Reading tissue metadata...")
    metadata_df = pd.read_csv(metadata_path, sep="\t", low_memory=False, usecols=["SAMPID", "SMTSD"])
    metadata_df = metadata_df[metadata_df["SAMPID"].str.startswith("GTEX-")]
    print(metadata_df.shape)
    metadata_df["GTEX_ID"] = metadata_df["SAMPID"].str.split("-").str[:2].str.join("-")
    gtex_tissue_dict = metadata_df.set_index("GTEX_ID")["SMTSD"].to_dict()

    sample_ids = df_index.str.split("-").str[:2].str.join("-")
    tissue_labels = sample_ids.map(gtex_tissue_dict).fillna("Unknown").tolist()

    if target_tissues is not None:
        tissue_labels = [label if label in target_tissues else "Other" for label in tissue_labels]
    
    tissue_to_organ = {
    # Brain
    "Brain - Cortex": "Brain",
    "Brain - Frontal Cortex (BA9)": "Brain",
    "Brain - Cerebellum": "Brain",
    "Brain - Cerebellar Hemisphere": "Brain",
    "Brain - Caudate (basal ganglia)": "Brain",
    "Brain - Nucleus accumbens (basal ganglia)": "Brain",
    "Brain - Putamen (basal ganglia)": "Brain",
    "Brain - Anterior cingulate cortex (BA24)": "Brain",
    "Brain - Spinal cord (cervical c-1)": "Brain",
    "Brain - Substantia nigra": "Brain",
    "Brain - Amygdala": "Brain",
    "Brain - Hippocampus": "Brain",
    "Brain - Hypothalamus": "Brain",

    # Adipose
    "Adipose - Subcutaneous": "Adipose Tissue",
    "Adipose - Visceral (Omentum)": "Adipose Tissue",

    # Artery
    "Artery - Aorta": "Artery",
    "Artery - Tibial": "Artery",
    "Artery - Coronary": "Artery",

    # Esophagus
    "Esophagus - Mucosa": "Esophagus",
    "Esophagus - Muscularis": "Esophagus",
    "Esophagus - Gastroesophageal Junction": "Esophagus",

    # Colon
    "Colon - Transverse": "Colon",
    "Colon - Sigmoid": "Colon",

    # Heart
    "Heart - Left Ventricle": "Heart",
    "Heart - Atrial Appendage": "Heart",

    # Skin
    "Skin - Sun Exposed (Lower leg)": "Skin",
    "Skin - Not Sun Exposed (Suprapubic)": "Skin",

    # Kidney
    "Kidney - Cortex": "Kidney",
    "Kidney - Medulla": "Kidney",

    # Cervix
    "Cervix - Ectocervix": "Cervix",
    "Cervix - Endocervix": "Cervix",

    # Lung
    "Lung": "Lung",

    # Muscle
    "Muscle - Skeletal": "Muscle",

    # Whole Blood
    "Whole Blood": "Blood",

    # Thyroid
    "Thyroid": "Thyroid",

    # Nerve
    "Nerve - Tibial": "Nerve",

    # Fibroblasts
    "Cells - Cultured fibroblasts": "Fibroblasts",

    # Lymphocytes
    "Cells - EBV-transformed lymphocytes": "Lymphocytes",

    # Breast
    "Breast - Mammary Tissue": "Breast",

    # Liver
    "Liver": "Liver",

    # Pancreas
    "Pancreas": "Pancreas",

    # Pituitary
    "Pituitary": "Pituitary",

    # Testis
    "Testis": "Testis",

    # Ovary
    "Ovary": "Ovary",

    # Prostate
    "Prostate": "Prostate",

    # Uterus
    "Uterus": "Uterus",

    # Vagina
    "Vagina": "Vagina",

    # Bladder
    "Bladder": "Bladder",

    # Stomach
    "Stomach": "Stomach",

    # Small Intestine
    "Small Intestine - Terminal Ileum": "Small Intestine",

    # Adrenal Gland
    "Adrenal Gland": "Adrenal",

    # Spleen
    "Spleen": "Spleen",

    # Minor Salivary Gland
    "Minor Salivary Gland": "Salivary Gland",

    # Fallopian Tube
    "Fallopian Tube": "Fallopian Tube",
}
    organ_labels = [tissue_to_organ.get(label, label) for label in tissue_labels]

    return organ_labels

def load_hgnc_protein_coding(filepath):
    print(f" [i] Reading HGNC protein coding gene...")
    df = pd.read_csv(filepath, sep="\t")
    df = df[df["Approved symbol"].str.startswith(("RPL", "RPS"), na = False)]
    df = df[df["Locus type"] == "gene with protein product"]
    df = df.dropna(subset=["Ensembl gene ID"]) # 確保 Ensembl gene ID 不是 NaN
    print(df)
    return dict(zip(df["Ensembl gene ID"], df["Approved symbol"]))

# 設定本地 HGNC 資料檔案
hgnc_file = r"~/gct_data/hgnc_protein_coding.tsv"
ensembl_to_symbol = load_hgnc_protein_coding(hgnc_file)
print(f" [i] Ribosomal Protein gene count: {len(ensembl_to_symbol)}")

def load_gct(filepath):
    """
    逐行解析 GCT 檔案，確保數據格式一致，避免記憶體使用過高。
    """
    print(" [1] Detecting column names...")
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_line = lines[2]  # `.gct` 的標題通常在第 3 行（index 2）
    column_names = header_line.strip().split("\t")
    gene_id_column = column_names[0]
    print(f" [2] Detected Gene_ID column: {gene_id_column}")

    print(" [3] Streaming GCT data...", end = "\n")
    print(" [i] This may take a few minutes, please wait...", end = "\n")
    
    original_gene_ids_all = []  # 收集所有 chunk 的原始索引（去除後綴）
    filtered_data = []
    
    def process_chunk(chunk):
        chunk = chunk.rename(columns={gene_id_column: "Gene_ID"})
        chunk["Gene_ID"] = chunk["Gene_ID"].astype(str)
        chunk.set_index("Gene_ID", inplace=True)

        # 🔹 確保基因 ID 的格式與 ensembl_to_symbol 一致（去掉 .X 後綴）
        chunk.index = chunk.index.str.split('.').str[0]
        
        # 🔹 儲存原始索引（不含後綴）
        original_index = chunk.index.astype(str).copy()
        original_gene_ids_all.extend(original_index)
        
        # 🔹 過濾我們要的 Ensembl ID
        if chunk.index is None or chunk.empty:
            return pd.DataFrame()

        return chunk[chunk.index.isin(ensembl_to_symbol)]

    chunk_size = 1000
    with pd.read_csv(filepath, sep="\t", skiprows=2, chunksize=chunk_size) as reader:
        for chunk in reader:
            processed_chunk = process_chunk(chunk)
            if not processed_chunk.empty:
                filtered_data.append(pa.Table.from_pandas(processed_chunk))

    print(" [4] Merging processed data...")
    final_table = pa.concat_tables(filtered_data).to_pandas()
    if "Description" in final_table.columns:
        final_table.drop(columns=["Description"], inplace=True)
    final_table = final_table.T  # 轉置以符合橫向基因矩陣的格式
    
    # 🔹 原始樣本名稱
    original_columns = final_table.columns.tolist()
    
    # 🔹 使用 LabelEncoder 進行樣本名稱編碼
    le = LabelEncoder()
    encoded_columns = le.fit_transform(original_columns)
    final_table.columns = encoded_columns
    
    # 🔹 建立還原對照表
    column_mapping = dict(zip(encoded_columns, original_columns))
    
    print(final_table.shape)
    
    # [Debug] 篩選 metadata 中 GTEx 的樣本
    # metadata_df = pd.read_csv(metadata_path, sep="\t", low_memory=False, usecols=["SAMPID", "SMTSD"])
    # metadata_df = metadata_df[metadata_df["SAMPID"].str.startswith("GTEX-")]

    # [Debug] 建立 SAMPID → SMTSD 的映射字典
    # sampid_to_tissue = metadata_df.set_index("SAMPID")["SMTSD"].to_dict()

    # [Debug] 從轉置後的 final_table 取出樣本 ID 並對應組織類別
    # sample_ids = final_table.index
    # organ_labels = pd.Series(sample_ids).map(sampid_to_tissue).fillna("Unknown").tolist()

    # [Optional] 印出組織分布確認
    # print(pd.Series(organ_labels).value_counts())
    
    return final_table, encoded_columns, column_mapping, original_gene_ids_all

# 預處理數據
def preprocess_data(df):
    print(" [5] Pre-processing data...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df = df.select_dtypes(include=[np.number])  # 只保留數值欄位
    data_scaled = scaler.fit_transform(df)
    return data_scaled, data_scaled.shape
# PCA 降維
def perform_pca_np(data, n_components=50):
    print(" [6] Performing PCA_NP...")
    data = data.astype(np.float32)
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    U, S, Vt = svds(data_centered, k=n_components)
    return np.dot(data_centered, Vt.T)
# t-SNE 降維
def perform_tsne_np(data, n_components=2, perplexity=30, learning_rate=500, max_iter=2000):
    print(" [7] Performing t-SNE_NP...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter)
    return tsne.fit_transform(data)
# 繪製 t-SNE 圖形
def plot_tsne(tsne_results, organ_labels, output_path):
    organ_order = [
    # 常見組織表達量多
    'Muscle', 'Blood', 'Skin', 'Adipose Tissue', 'Artery',
    'Thyroid', 'Nerve', 'Fibroblasts', 'Esophagus', 'Lung',
    'Colon', 'Heart', 'Testis', 'Stomach', 'Breast',
    'Pancreas', 'Lymphocytes', 'Pituitary', 'Brain', 'Adrenal',

    # 生殖系統（男性、女性）
    'Ovary', 'Uterus', 'Vagina', 'Prostate',

    # 消化與排泄
    'Liver', 'Small Intestine', 'Kidney', 'Bladder',

    # 感官與內分泌小腺體
    'Salivary Gland', 'Spleen', 'Fallopian Tube'
]
    # 建立 HLS 色盤（自動環狀配色）
    palette = sns.hls_palette(n_colors=len(organ_order), l=0.65, s=0.8)

    # 將器官順序與顏色對應為 dict
    organ_color_dict = dict(zip(organ_order, palette))
    
    print(" [8] Drawing t-SNE plot...")
    plt.figure(figsize=(19.2, 10.8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=organ_labels, palette=organ_color_dict, s=50, edgecolor="black", alpha=0.8)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE visualization of GTEx Data")
    plt.legend(title="Tissues", bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()  # 確保圖例不會超出畫布
    plt.savefig(output_path, dpi=300)
    print(f" [i] t-SNE figure is saved as: {output_path}")

# 主執行流程
if __name__ == "__main__":
    start = time.time()
    gct_file = r"/home/terry_0714/gct_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
    metadata_path = r"/home/terry_0714/gct_data/GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv"
    hgnc_file = r"~/gct_data/hgnc_protein_coding.tsv"
    output_image = r"/home/terry_0714/tsne_plot/tsne_plot_new_13.png"
    # 讀取 GCT 檔案
    gct_df, encoded_columns, column_mapping, original_gene_ids_all = load_gct(gct_file)
    organ_labels = load_metadata_labels(metadata_path, gct_df.index)
    
    # 資料預處理
    processed_data, shape = preprocess_data(gct_df)
    
    # 資料降維
    pca_data = perform_pca_np(processed_data)
    
    # t-SNE 降維
    tsne_results = perform_tsne_np(pca_data)
    
    # 作圖
    # # Optional: 若只想視覺化出現最多的前 10 種組織，可取消以下註解
    # top_10_tissues = pd.Series(organ_labels).value_counts().nlargest(10).index.tolist()
    # filtered_idx = [i for i, label in enumerate(organ_labels) if label in top_10_tissues]
    # tsne_results = tsne_results[filtered_idx]
    # organ_labels = [organ_labels[i] for i in filtered_idx]
    
    plot_tsne(tsne_results, organ_labels, output_image)
    print(" [i] t-SNE analysis is completed.")

    end = time.time()
    elapsed_time = int(end - start)
    hours = elapsed_time // 3600
    minutes = elapsed_time % 3600 // 60
    seconds = elapsed_time % 60

    print(f" [i] Time spent: {hours} hours, {minutes} minutes and {seconds} seconds.")