import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Ana veri setinin yolu
data_dir = "C:\\Users\\yldrm\\.cache\\kagglehub\\datasets\\harieh\\ocr-dataset\\versions\\1\\dataset"  # Buraya veri setinizin ana klasör yolunu yazın

# Çıktı klasörleri (train, test, validation)
output_dir = "split_dataset"  # Bölünmüş verilerin kaydedileceği ana klasör
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Bölme oranları (train, test, validation)
train_ratio = 0.7
test_ratio = 0.22
val_ratio = 0.08

# Her bir sınıf için işlem yap
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    if not os.path.isdir(class_dir):
        continue
    
    # Sınıfın tüm dosyalarını listele
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    
    # Dosyaları rastgele böl
    train_files, temp_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)
    test_files, val_files = train_test_split(temp_files, test_size=val_ratio/(test_ratio + val_ratio), random_state=42)
    
    # Klasörleri oluştur ve dosyaları kopyala
    for split, files_split in [("train", train_files), ("test", test_files), ("val", val_files)]:
        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        
        for file in files_split:
            src_path = os.path.join(class_dir, file)
            dst_path = os.path.join(split_class_dir, file)
            shutil.copy(src_path, dst_path)
    
    print(f"Processed class: {class_name}")

print("Dataset splitting completed successfully!")