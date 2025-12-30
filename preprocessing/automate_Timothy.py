import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# KONFIGURASI PATH
# Pastikan folder credit_risk_raw ada di luar folder preprocessing
RAW_DATA_PATH = os.path.join('..', 'credit_risk_raw', 'credit_risk_dataset.csv')
OUTPUT_FILE = 'credit_risk_preprocessing.csv'

# FUNGSI MEMUAT DATA
def load_data(path):
    print(f"📂 Mencoba memuat data mentah dari: {path}")
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Dataset berhasil dimuat! Dimensi: {df.shape}")
        return df
    else:
        print(f"❌ ERROR: File tidak ditemukan di {path}")
        exit()

# FUNGSI PREPROCESSING
def preprocess_data(df):
    print("⚙️ Memulai proses Preprocessing...")
    
    # HAPUS DUPLIKAT
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    duplikat = initial_len - len(df)
    
    if duplikat > 0:
        print(f"✅ Berhasil menghapus {duplikat} data duplikat")
    else:
        print("✅ Tidak ada data duplikat ditemukan")

    # HANDLING MISSING VALUES
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # ISI NUMERIK DENGAN MEDIAN
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        
    # ISI KATEGORIKAL DENGAN MODUS
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("✅ Missing values berhasil diisi")

    # ENCODING
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    print("✅ Encoding selesai")
    
    return df

# EKSEKUSI UTAMA
if __name__ == "__main__":
    print("🚀 MULAI OTOMASI DATA")
    
    try:
        # LOAD DATA
        df = load_data(RAW_DATA_PATH)
        
        # PROSES DATA
        df_clean = preprocess_data(df)
        
        # SIMPAN HASIL
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print(f"🎉 SUKSES! Data bersih disimpan sebagai: {OUTPUT_FILE}")
        print("📂 File ini siap digunakan untuk training model")
            
    except Exception as e:
        print(f"❌ TERJADI KESALAHAN: {e}")