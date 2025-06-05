import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from PIL import Image
import cv2
import random


# === Aynı model yapısı ===
class OCRCNN(nn.Module):
    def __init__(self, num_classes):
        super(OCRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# === Görsel işleme ===
def advanced_preprocessing(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return Image.fromarray(gray)

# === Test veri dizini ===
test_path = "C:\\Users\\yldrm\\VSCodeProjects\\YZ\\split_dataset\\test"

transform = transforms.Compose([
    transforms.Lambda(advanced_preprocessing),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Modeli yükle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(test_dataset.classes)
model = OCRCNN(num_classes).to(device)
model.load_state_dict(torch.load("ocr_cnn_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# === Test işlemi ===
all_preds = []
all_labels = []
incorrect_samples = []

test_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Hatalı tahminleri sakla
        for img, pred, label in zip(images, preds, labels):
            if pred != label and len(incorrect_samples) < 5:
                incorrect_samples.append((img.cpu(), pred.item(), label.item()))

accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
avg_loss = test_loss / len(test_loader)

print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
print(f"❌ Average Test Loss: {avg_loss:.4f}")

# === Raporlar ===
print("\n📊 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("🧩 Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === Yanlış tahmin edilen bazı görselleri göster ===
def imshow(img_tensor, mean=0.5, std=0.5):
    img = img_tensor.squeeze().numpy()
    img = img * std + mean  # Unnormalize
    plt.imshow(img, cmap='gray')

print("\n🖼️ Incorrect Predictions:")
plt.figure(figsize=(12, 6))
for idx, (img, pred, label) in enumerate(incorrect_samples):
    plt.subplot(1, 5, idx + 1)
    imshow(img)
    plt.title(f"P: {test_dataset.classes[pred]}\nA: {test_dataset.classes[label]}")
    plt.axis("off")
plt.suptitle("Some Misclassified Test Samples")
plt.tight_layout()
plt.show()



# === Rastgele bir görsel seç ve tahmin yap ===
def predict_random_sample():
    # Dataset içinden rastgele bir örnek al
    rand_idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[rand_idx]
    image_input = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_input)
        _, predicted = torch.max(output, 1)

    # Görseli göster
    plt.figure(figsize=(3, 3))
    imshow(image.cpu())
    plt.title(f"Gerçek: {test_dataset.classes[label]}\nTahmin: {test_dataset.classes[predicted.item()]}")
    plt.axis("off")
    plt.suptitle("🎯 Model Tahmini (Rastgele Test Görseli)")
    plt.tight_layout()
    plt.show()


# === Kullanıcıdan giriş alarak döngü ===
while True:
    komut = input("\nYeni tahmin için '1' yaz, çıkmak için '0' yaz: ")
    if komut == "1":
        predict_random_sample()
    elif komut == "0":
        print("👋 Tahmin döngüsü sonlandırıldı.")
        break
    else:
        print("⚠️ Sadece '1' veya '0' giriniz.")