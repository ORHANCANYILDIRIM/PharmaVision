import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
import cv2


# === MODELİN AYNI OLMASI GEREKİYOR ===
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


# === Advanced Preprocessing ===
def advanced_preprocessing(img):
    img = np.array(img)

    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv2.resize(img, (128, 128))
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    return Image.fromarray(gray)


# === Veri yolu ve dönüşüm tanımı ===
val_path = "C:\\Users\\yldrm\\VSCodeProjects\\YZ\\split_dataset\\val"

transform = transforms.Compose([
    transforms.Lambda(advanced_preprocessing),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Val dataset ve loader ===
val_dataset = ImageFolder(root=val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# === Cihaz ayarı ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Modeli oluştur ve yükle ===
num_classes = len(val_dataset.classes)
model = OCRCNN(num_classes).to(device)
model.load_state_dict(torch.load("ocr_cnn_model.pth", map_location=device))
model.eval()

# === Loss fonksiyonu (opsiyonel) ===
criterion = nn.CrossEntropyLoss()

# === Validasyon işlemi ===
correct = 0
total = 0
val_loss = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_loss = val_loss / len(val_loader)

print(f"\n📊 Validation Accuracy: {accuracy:.2f}%")
print(f"📉 Average Validation Loss: {avg_loss:.4f}")
