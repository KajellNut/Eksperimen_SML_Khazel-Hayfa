import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean_data():
    print("Memulai proses Data Preprocessing otomatis...")

    # Load Data
    input_file = 'D:\Kuliah Kajel\Asah Dicoding\Submission\MSML\E Commerce Dataset_raw.xlsx'

    # Cek path agar aman (biar bisa jalan dari folder mana saja)
    if not os.path.exists(input_file):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, 'E Commerce Dataset_raw.xlsx')

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' tidak ditemukan.")
        return

    df = pd.read_excel(input_file, sheet_name='E Comm')

    # Handling Missing Values
    # Isi kolom angka dengan Median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Isi kolom teks dengan Modus
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if len(df[col].mode()) > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Handling Outlier (Metode Capping/Winsorizing)
    # Ambil kolom numerik, tanpa 'Churn' (Target) dan 'CustomerID'
    cols_to_clean = [col for col in numeric_cols if col not in ['Churn', 'CustomerID']]
    
    for feature in cols_to_clean:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        # Ganti nilai yang terlalu tinggi/rendah dengan batasnya
        df[feature] = np.where(df[feature] > upper_limit, upper_limit, df[feature])
        df[feature] = np.where(df[feature] < lower_limit, lower_limit, df[feature])

    # Encoding (Mengubah semua Teks menjadi Angka)
    le = LabelEncoder()
    # Update daftar kolom object
    obj_cols = df.select_dtypes(include=['object']).columns
    
    for col in obj_cols:
        df[col] = le.fit_transform(df[col])

    # Simpan Data Bersih ke Folder 'Membangun_model'
    output_folder = 'D:\Kuliah Kajel\Asah Dicoding\Submission\MSML\Preprocessing'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_filename = os.path.join(output_folder, 'E Commerce Dataset_preprocessing.csv')
    
    # Hapus kolom CustomerID sebelum disimpan karena tidak dipakai training
    if 'CustomerID' in df.columns:
        df.drop('CustomerID', axis=1, inplace=True)

    df.to_csv(output_filename, index=False)
    print(f"SUKSES: Data bersih disimpan di '{output_filename}'")

if __name__ == "__main__":
    load_and_clean_data()