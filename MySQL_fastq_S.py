import pymysql
from Bio import SeqIO
import time

start = time.time()

# 連接 MySQL
conn = pymysql.connect(
        unix_socket="/var/run/mysqld/mysqld.sock",
        user="dbeaver",
        password="nTHU_110081033",
        database="bio_db",
        port=3306
)
cursor = conn.cursor()

print("✅ 已建立與MySQL資料庫的連線")

# 建立 HTS 數據表（若尚未建立）
cursor.execute("""
    CREATE TABLE IF NOT EXISTS fastq_data (
        read_id VARCHAR(50) PRIMARY KEY,
        sequence TEXT,
        quality_score TEXT,
        read_length INT,
        file_source VARCHAR(100)
    )
""")

print("✅ 已建立 HTS 數據表")

# 讀取 FASTQ 並存入 MySQL
fastq_file = "/home/terry_0714/hts_data/SRR14485880_1.fastq"  # 替換為你的 FASTQ 檔案
batch_size = 10000  # 設定批次大小
batch_data = []

print("▶️ 正在將 FASTQ 數據存入 MySQL...")

for record in SeqIO.parse(fastq_file, "fastq"):
    read_id = record.id
    sequence = str(record.seq)
    quality_score = ",".join(map(str, record.letter_annotations["phred_quality"]))
    read_length = len(record.seq)
    batch_data.append((read_id, sequence, quality_score, read_length, fastq_file))
    
    # 當達到批次大小時，執行批次 INSERT
    if len(batch_data) >= batch_size:
        query = """
            INSERT INTO fastq_data (read_id, sequence, quality_score, read_length, file_source) 
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            sequence=VALUES(sequence), quality_score=VALUES(quality_score), 
            read_length=VALUES(read_length), file_source=VALUES(file_source)
        """
        print(f" [訊息] 本批次已正常寫入MySQL ")
        cursor.executemany(query, batch_data)
        conn.commit()
        batch_data = []  # 清空批次數據

# 插入最後一批數據
if batch_data:
    cursor.executemany(query, batch_data)
    conn.commit()

cursor.close()
conn.close()
print("✅ FASTQ 數據已存入 MySQL")

end = time.time()
elapsed_time = int(end - start)
hours = elapsed_time // 3600
minutes = elapsed_time % 3600 // 60
seconds = elapsed_time % 60

print(f"Time spent: {hours} hours, {minutes} minutes and {seconds} seconds.")