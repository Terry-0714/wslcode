import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
import pyarrow as pa
import pyarrow.compute as pc

start = time.time()

print(f" [i] Start analyzing, total 15 steps...")
print(f" [i] Setting up...")

# 強制 Qt 在 Linux 環境下使用 X11
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 防止 NumPy 佔用過多記憶體
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 固定隨機種子確保可重現性
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# 讀取 HGNC 蛋白質編碼基因對應表
def load_hgnc_protein_coding(filepath):
    df = pd.read_csv(filepath, sep="\t")
    # Debugging: Ensuring columns' names 
    # print(df.head())
    df = df[df["Locus type"] == "gene with protein product"]
    df = df.dropna(subset=["Ensembl gene ID"]) # 確保 Ensembl gene ID 不是 NaN
    df["Ensembl gene ID"] = df["Ensembl gene ID"].astype(str) # 確保 Ensembl gene ID 是字串
    return dict(zip(df["Ensembl gene ID"], df["Approved symbol"]))

# 設定本地 HGNC 資料檔案
print(f" [1] Reading HGNC protein coding gene...")
hgnc_file = r"~/gct_data/hgnc_protein_coding.tsv"
ensembl_to_symbol = load_hgnc_protein_coding(hgnc_file)
# Debugging: 確認 ensembl_to_symbol 字典不為空
# print(ensembl_to_symbol)
# 選擇對比組織
target_tissues = ["Brain - Cortex", "Liver", "Muscle - Skeletal", "Whole Blood", "Skin - Sun Exposed (Lower leg)"]

def load_gct(filepath, metadata_path):
    """
    逐行解析 GCT 檔案，確保數據格式一致，避免記憶體使用過高。
    """
    print(" [2] Reading metadata...")
    metadata_df = pd.read_csv(metadata_path, sep="\t", low_memory=False, usecols=["SAMPID", "SMTSD"])
    metadata_df["GTEX_ID"] = metadata_df["SAMPID"].str.split("-").str[:2].str.join("-")
    
    # 只保留對比組織 
    metadata_df = metadata_df[metadata_df["SMTSD"].isin(target_tissues)]
    # Debugging: See the columns
    # print(metadata_df)
    gtex_tissue_dict = metadata_df.set_index("GTEX_ID")["SMTSD"].to_dict()

    print(" [3] Selecting tissues...")
    filtered_samples = set(metadata_df[metadata_df["SMTSD"].isin(target_tissues)]["SAMPID"])

    print(" [4] Detecting column names...")
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_line = lines[2]  # `.gct` 的標題通常在第 3 行（index 2）
    column_names = header_line.strip().split("\t")
    gene_id_column = column_names[0]
    print(f" [5] Detected Gene_ID column: {gene_id_column}")

    print(" [6] Streaming GCT data...", end = "\n")
    print(" [i] This may take a few minutes, please wait...", end = "\n")
    def process_chunk(chunk):
        chunk = chunk.rename(columns={gene_id_column: "Gene_ID"})
        chunk["Gene_ID"] = chunk["Gene_ID"].astype(str)
        chunk.set_index("Gene_ID", inplace=True)
        num_cols = [col for col in chunk.columns if col in filtered_samples]
        chunk[num_cols] = chunk[num_cols].astype("float32", errors="ignore")
        return chunk[num_cols]

    chunk_size = 10000
    filtered_data = []
    gene_ids = []

    with pd.read_csv(filepath, sep="\t", skiprows=2, chunksize=chunk_size) as reader:
        for chunk in reader:
            processed_chunk = process_chunk(chunk)
            filtered_data.append(pa.Table.from_pandas(processed_chunk))
            gene_ids.extend(processed_chunk.index.tolist())

    print(" [7] Merging processed data...")
    final_table = pa.concat_tables(filtered_data)

    print(" [8] Calculating standard errors efficiently...")
    numeric_columns = [col for col in final_table.column_names if col != "Gene_ID"]
    numeric_table = final_table.select(numeric_columns)
    schema = pa.schema([(col, pa.float32()) for col in numeric_columns])
    numeric_table = numeric_table.cast(schema)
    numeric_df = numeric_table.to_pandas()

    gene_variability = numeric_df.std(axis=1)
    print(" [9] Selecting top 100 variable genes efficiently...")
    gene_variability_array = pa.array(gene_variability)
    sorted_indices = pc.sort_indices(gene_variability_array, sort_keys=[("values", "descending")])[:100]
    top_variable_genes = [gene_ids[i.as_py()] for i in sorted_indices]
    top_variable_genes = [gene.split(".")[0] for gene in top_variable_genes]
    # Debugging
    # print(top_variable_genes)
    print(f" [10] Filtering protein-coding genes using HGNC data...")
    final_genes = [gene for gene in top_variable_genes if gene in ensembl_to_symbol]
    # Debugging
    # print(final_genes)
    print(f" [11] Filtering highly expressed genes...")
    df = final_table.to_pandas()
    df = df.T  # 轉置數據，使樣本成為行，基因成為列
    df.columns = df.columns.str.split('.').str[0]  # 去除 Ensembl ID 後綴
    # Debugging: Ensure the identity
    """
    print(f"🚀 GCT 檔案中的基因數量: {len(df.columns)}")
    print(f"🚀 前 5 個基因: {df.columns[:5]}")
    print(f"🚀 最後 5 個基因: {df.columns[-5:]}")
    print(f"🚀 篩選後的基因數量: {len(final_genes)}")
    print(f"🚀 前 5 個篩選基因: {final_genes[:5]}")
    """
    df = df.loc[:, final_genes]  # 過濾出高變異度基因
    
    sample_ids = df.index.str.split("-").str[:2].str.join("-")
    tissue_labels = sample_ids.map(gtex_tissue_dict).fillna("Unknown").tolist()
    
    print(f" [i] DataFrame 形狀: {df.shape}")
    df = df.select_dtypes(include=[np.number])  # 只保留數值型資料
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df, tissue_labels

# 預處理數據
def preprocess_data(df):
    print(" [12] Pre-processing data...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled, data_scaled.shape

# PCA 降維
def perform_pca_np(data, n_components=50):
    print(" [13] Performing PCA_NP...")
    data = data.astype(np.float32)
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    U, S, Vt = svds(data_centered, k=n_components)
    return np.dot(data_centered, Vt.T)

# t-SNE 降維
def perform_tsne_np(data, n_components=2, perplexity=30, learning_rate=500, max_iter=2000):
    print(" [14] Performing t-SNE_NP...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter)
    return tsne.fit_transform(data)

# 繪製 t-SNE 圖形
def plot_tsne(tsne_results, tissue_labels, output_path):
    print(" [15] Drawing t-SNE plot...")
    plt.figure(figsize=(19.2, 10.8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=tissue_labels, palette='Set2', s=50, edgecolor="black", alpha=0.8)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE visualization of GTEx Data")
    plt.legend(title="Tissues", bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()  # 確保圖例不會超出畫布
    plt.savefig(output_path, dpi=300)
    print(f" [i] t-SNE figure is saved as: {output_path}")

# 主執行流程
if __name__ == "__main__":
    gct_file = r"/home/terry_0714/gct_data/GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_tpm_non_lcm.gct"
    metadata_path = r"/home/terry_0714/gct_data/GTEx_Analysis_v10_Annotations_SampleAttributesDS.tsv"
    output_image = r"/home/terry_0714/tsne_plot/tsne_plot_new_4.png"

    df, sample_labels = load_gct(gct_file, metadata_path)
    
    processed_data, shape = preprocess_data(df)
    
    pca_data = perform_pca_np(processed_data)
    
    tsne_results = perform_tsne_np(pca_data)
    
    plot_tsne(tsne_results, sample_labels, output_image)
    
    print(" [i] t-SNE analysis is completed.")

    end = time.time()
    elapsed_time = int(end - start)
    hours = elapsed_time // 3600
    minutes = elapsed_time % 3600 // 60
    seconds = elapsed_time % 60

    print(f" [i] Time spent: {hours} hours, {minutes} minutes and {seconds} seconds.")