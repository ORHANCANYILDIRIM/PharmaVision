import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# === Cihaz kontrolü ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model tanımı ===
class OCRCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(OCRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# === Görsel ön işleme ===
def advanced_preprocessing(img):
    img = np.array(img)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv2.resize(img, (128, 128))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(img)
    return Image.fromarray(gray)

# === Transform işlemleri ===
transform = transforms.Compose([
    transforms.Lambda(advanced_preprocessing),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Model ve sınıf isimlerini yükle ===
num_classes = 36
model = OCRCNN(num_classes).to(device)
model.load_state_dict(torch.load("models/ocr_cnn_model.pth", map_location=device))
model.eval()

train_path = "split_dataset/train"
class_names = sorted([cls for cls in os.listdir(train_path) if not cls.startswith(".")])

# === Karakterleri ayıklama (kutu bilgisi dahil) ===
def extract_characters(image_path):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height = gray.shape[0]

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)

        if w > 10 and h > 10 and area > 300 and 0.2 < aspect_ratio < 1.2:
            boxes.append((x, y, w, h))

    if not boxes:
        return [], original

    # Satıra göre grupla
    rows = []
    boxes = sorted(boxes, key=lambda b: b[1])

    for box in boxes:
        x, y, w, h = box
        matched = False
        for group in rows:
            gx, gy, gw, gh = group[0]
            if abs(y - gy) < 20:
                group.append(box)
                matched = True
                break
        if not matched:
            rows.append([box])

    selected_row = max(rows, key=lambda group: np.mean([h for _, _, _, h in group]))

    bounding_boxes = []
    for x, y, w, h in sorted(selected_row, key=lambda b: b[0]):
        roi = gray[y:y+h, x:x+w]
        roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        roi_resized = cv2.resize(roi, (32, 32))
        pil_image = Image.fromarray(roi_resized)
        bounding_boxes.append((pil_image, (x, y, w, h)))

    return bounding_boxes, original

# === Tahmin fonksiyonu + kutu çizimi ===
def predict_and_visualize(image_path):
    characters, original = extract_characters(image_path)
    word = ""

    for idx, (char_img, (x, y, w, h)) in enumerate(characters):
        processed = transform(char_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(processed)
            _, pred = torch.max(output, 1)
        predicted_char = class_names[pred.item()]
        word += predicted_char

        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original, predicted_char, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"\n🔤 Tahmin Edilen Kelime: {word}")

    # Görseli göster
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Tahmin: {word}")
    plt.show()

# === Test çalıştırıcı ===
if __name__ == "__main__":
    image_path = input("Görselin yolunu girin: ")
    predict_and_visualize(image_path)
