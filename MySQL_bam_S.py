import pysam
import pymysql
import time

start = time.time()

# 連接 MySQL
try:
    conn = pymysql.connect(
        unix_socket="/var/run/mysqld/mysqld.sock",
        user="dbeaver",
        password="nTHU_110081033",
        database="bio_db",
        port=3306
    )
    print("✅ 已建立與MySQL資料庫的連線")

except pymysql.Error as err:
    print(f"錯誤：{err}")
    exit()

else:
    cursor = conn.cursor()

    # 建立 HTS 數據表（若尚未建立）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bam_data (
            read_id VARCHAR(50) PRIMARY KEY,
            sequence TEXT,
            mapping_quality INT,
            read_length INT,
            file_source VARCHAR(100)
        )
    """)

    print("✅ 已建立 HTS 數據表")

    # 讀取 BAM 並存入 MySQL
    print("▶️  正在將 BAM 數據存入 MySQL...")

    bam_file = "/home/terry_0714/hts_data/SRR14485880_sorted.bam"
    batch_size = 10000  # 設定批次大小
    batch_data = []

    with pysam.AlignmentFile(bam_file, "rb", check_sq=True) as bam:
        for read in bam.fetch():
            read_id = read.query_name
            sequence = read.query_sequence
            mapping_quality = read.mapping_quality
            read_length = len(sequence)
            batch_data.append((read_id, sequence, mapping_quality, read_length, bam_file))
            
            # 當達到批次大小時，執行批次 INSERT
            if len(batch_data) >= batch_size:
                query = """
                    INSERT INTO bam_data (read_id, sequence, mapping_quality, read_length, file_source) 
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    sequence=VALUES(sequence), mapping_quality=VALUES(mapping_quality), 
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
    print("✅ BAM 數據已存入 MySQL")

    end = time.time()
    elapsed_time = int(end - start)
    hours = elapsed_time // 3600
    minutes = elapsed_time % 3600 // 60
    seconds = elapsed_time % 60

    print(f"Time spent: {hours} hours, {minutes} minutes and {seconds} seconds.")
