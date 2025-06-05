import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import os


print("1")
# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("2")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA kullanılıyor. Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA versiyon: {torch.version.cuda}")
    print(f"PyTorch CUDA versiyon: {torch.backends.cudnn.version()}")
else:
    device = torch.device("cpu")
    print("CUDA kullanılamıyor, CPU kullanılıyor.")


u=0
# Ön işleme fonksiyonu
def advanced_preprocessing(img):
    global u
    img = np.array(img)
    
    # 1. Hızlı boyut küçültme
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv2.resize(img, (128, 128))
    
    # 2. Gri tonlama
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 3. Basit kontrast artırma
    gray = cv2.equalizeHist(gray)

    print(u)
    u= u + 1
    return Image.fromarray(gray)
print("3")

# Dönüşümler
transform = transforms.Compose([
    transforms.Lambda(advanced_preprocessing),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
print("4")

# Sadeleştirilmiş CustomImageFolder
class DirectLabelImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        """Klasör isimlerini direkt sınıf etiketi olarak kullanır"""
        classes = sorted([d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d))])
        if not classes:
            raise FileNotFoundError(f"No classes found in {directory}")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"Bulunan sınıflar: {classes}")
        return classes, class_to_idx
print("5")

# Veri yolları
data_root = "C:\\Users\\yldrm\\VSCodeProjects\\YZ\\split_dataset"
train_path = os.path.join(data_root, "train")
val_path = os.path.join(data_root, "val")
test_path = os.path.join(data_root, "test")
print("6")

# Veri setlerini yükle
try:
    print("Veri setleri yükleniyor...")
    train_dataset = DirectLabelImageFolder(root=train_path, transform=transform)
    val_dataset = DirectLabelImageFolder(root=val_path, transform=transform)
    test_dataset = DirectLabelImageFolder(root=test_path, transform=transform)
    
    print(f"\nYükleme başarılı!")
    print(f"Train örnek sayısı: {len(train_dataset)}")
    print(f"Validasyon örnek sayısı: {len(val_dataset)}")
    print(f"Test örnek sayısı: {len(test_dataset)}")
    print(f"Sınıflar: {train_dataset.classes}")


except Exception as e:
    print(f"\n❌ Hata: {str(e)}")
    # Hata ayıklama için ek bilgi
    sample_class = "A"
    sample_path = os.path.join(train_path, sample_class)
    print(f"\nDebug info for {sample_class}:")
    print(f"Klasör var mı: {os.path.exists(sample_path)}")
    if os.path.exists(sample_path):
        print(f"Dosyalar: {os.listdir(sample_path)[:5]}")  # İlk 5 dosya
        if os.listdir(sample_path):
            sample_file = os.path.join(sample_path, os.listdir(sample_path)[0])
            print(f"Örnek dosya boyutu: {os.path.getsize(sample_file)} bayt")
print("7")

# DataLoader'lar
if 'train_dataset' in locals():
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("8")

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("9")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("10")

# CNN Modeli
class OCRCNN(nn.Module):
    print("modelin içinde")
    def __init__(self, num_classes):
        print("modelin içinde1")

        super(OCRCNN, self).__init__()
        print("modelin içinde3")

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Model oluşturma
num_classes = len(train_dataset.classes)
print(f"Toplam {num_classes} sınıf bulundu: {train_dataset.classes}")

model = OCRCNN(num_classes).to(device)

# Kayıp fonksiyonu ve optimizer
i_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model eğitimi
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = i_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

print("✅ Model eğitimi tamamlandı!")

# Modeli kaydetme
torch.save(model.state_dict(), "ocr_cnn_model.pth")
print("✅ Model kaydedildi: ocr_cnn_model.pth")
