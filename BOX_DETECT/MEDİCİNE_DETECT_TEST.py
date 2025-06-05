from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Eğitilmiş modeli yükle
model = YOLO("C:\\Users\\yldrm\\Documents\\GitHub\\proje-kod-ve-raporu-ORHANCANYILDIRIM\\runs\\detect\\train19\\weights\\best.pt")

# Test görüntüsünü yükle
image_path = "C:\\Users\\yldrm\\Documents\\GitHub\\proje-kod-ve-raporu-ORHANCANYILDIRIM\\Kutu Tespiti\\Medicine-box-1\\test\\images\\1-32-_jpg.rf.c225f09c2172e80ef58ca4e8e876f8cb.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatını RGB'ye çevir

# Tahmin yap
results = model.predict(source=image_path, conf=0.5, save=True, show=True)

# Tahmin sonuçlarını al
result = results[0]
boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box koordinatları
classes = result.boxes.cls.cpu().numpy()  # Sınıf ID'leri
confidences = result.boxes.conf.cpu().numpy()  # Güven skorları
class_names = model.names  # Sınıf isimleri sözlüğü

# Görselleştirme
plt.figure(figsize=(12, 8))
plt.imshow(image)
ax = plt.gca()

# Her bir tespit edilen nesne için
for box, cls, conf in zip(boxes, classes, confidences):
    # Bounding box çiz
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    
    # Sınıf ismi ve güven skoru ekle
    label = f"{class_names[int(cls)]}: {conf:.2f}"
    ax.text(x1, y1 - 10, label, color='red', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

plt.axis('off')
plt.title('YOLO Tahmin Sonuçları', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Konsola bilgileri yazdır
print("\nDetaylı Tahmin Bilgileri:")
for i, (cls, conf) in enumerate(zip(classes, confidences)):
    print(f"Nesne {i+1}:")
    print(f"  Sınıf: {class_names[int(cls)]} (ID: {int(cls)})")
    print(f"  Güven Skoru: {conf:.4f}")
    print(f"  Bounding Box: {boxes[i]}")
    print("-" * 40)