import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# === Cihaz kontrolü ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Ön işleme fonksiyonu ===
def advanced_preprocessing(img):
    img = np.array(img)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return Image.fromarray(gray)

# === Dönüşümler ===
transform = transforms.Compose([
    transforms.Lambda(advanced_preprocessing),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Model tanımı ===
class OCRCNN(nn.Module):
    def __init__(self, num_classes):
        super(OCRCNN, self).__init__()
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

# === Modeli yükleme ===
num_classes = 36  # Örnek olarak harfler ve rakamlar
model = OCRCNN(num_classes).to(device)
model.load_state_dict(torch.load("models//ocr_cnn_model.pth", map_location=device))
model.eval()

import os

# Eğitim veri kümesinin yolu (Eğitim sırasında kullanılan klasör)
train_path = "split_dataset/train"  # Bunu kendi eğitim veri yolunla değiştir!

# Klasörlerden sınıf isimlerini al
class_names = sorted(os.listdir(train_path))

# Eğer gizli dosyalar varsa (örneğin .DS_Store), filtrele
class_names = [cls for cls in class_names if not cls.startswith(".")]

print("Sınıf Etiketleri:", class_names)


# === Görsel tahmin fonksiyonu ===
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Hata: Dosya bulunamadı!")
        return
    
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(processed_image)
        _, predicted = torch.max(output, 1)
    
    predicted_label = class_names[predicted.item()]
    print(f"Tahmin Edilen Karakter: {predicted_label}")


    
    plt.imshow(image)
    plt.title(f"Tahmin: {predicted.item()}")
    plt.axis("off")
    plt.show()

# === Kullanıcıdan giriş al ===
if __name__ == "__main__":
    image_path = input("Lütfen test etmek istediğiniz görselin yolunu girin: ")
    predict_image(image_path)
print(num_classes)
