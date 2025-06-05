from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import os
import uuid
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from BACKEND.model import predict_word
import traceback
from rapidfuzz import process
import math

#        uvicorn BACKEND.app_model:app --reload


app = FastAPI()

# CORS ayarı (React ile bağlantı için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO modelini yükle (kutu tespiti için)
yolo_model = YOLO("MODELS/kutu_modeli.pt")

# === CSV dosyasından ilaç bilgisi bulma fonksiyonu ===
# Yardımcı: JSON uyumlu hale getir
def sanitize_dict(d):
    def safe(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "Bilinmiyor"
        return v
    return {k: safe(v) for k, v in d.items()}

# === CSV dosyasından ilaç bilgisi bulma fonksiyonu ===
def get_medicine_info_from_csv(medicine_name):
    csv_path = "data/medicine_info.csv"
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    # Tüm ilaç isimlerini listele
    all_medicines = df["medicine_name"].tolist()

    # Tahmin edilen kelime ile en benzer eşleşmeyi bul
    best_match, score, _ = process.extractOne(medicine_name, all_medicines)

    print(f"🔍 En iyi eşleşme: {best_match} (Skor: {score})")


    if score < 70:
        return {"Bilgi": "Benzer bir kayıt bulunamadı."}
    
    result = df[df["medicine_name"].str.upper() == best_match.upper()]
    if not result.empty:
        return sanitize_dict(result.iloc[0].to_dict())

    return {"Bilgi": "Benzer kayıt bulunamadı."}


# === Ana API endpoint'i ===
@app.post("/kutu-tahmin")
async def tahmin_kutudan(file: UploadFile = File(...)):
    try:
        print("📥 Görsel yükleniyor...")
        image_id = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join("temp", image_id)
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(await file.read())

        print(f"📂 Görsel kaydedildi: {image_path}")

        print("📦 YOLO ile kutu tespiti başlatılıyor...")
        results = yolo_model.predict(source=image_path, conf=0.4, save=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            print("❗ Kutu tespit edilemedi, tüm görsel OCR işlemine gönderilecek.")
            kelime = predict_word(image_path)  # ← Artık use_full_image yok
            os.remove(image_path)
            print(f"🔤 OCR sonucu (tüm görsel): {kelime}")
            return {
                "tahmin": kelime,
                "ilac_bilgisi": get_medicine_info_from_csv(kelime) or "Bilgi bulunamadı.",
                "uyari": "Kutu tespiti yapılamadı, tüm görselden okuma yapıldı."
            }

        print("✅ Kutu tespiti başarılı. İlk kutu kırpılıyor...")
        xyxy = boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        img = cv2.imread(image_path)
        cropped = img[y1:y2, x1:x2]
        cropped_path = os.path.join("temp", "cropped_" + image_id)
        cv2.imwrite(cropped_path, cropped)

        print("🔤 Kırpılan görüntü OCR işlemine gönderiliyor...")
        kelime = predict_word(cropped_path)
        print(f"📘 OCR sonucu: {kelime}")
        ilac_bilgisi = get_medicine_info_from_csv(kelime)
        print("📄 CSV araması tamamlandı.")
        os.remove(image_path)
        os.remove(cropped_path)

        return {
            "tahmin": kelime,
            "ilac_bilgisi": ilac_bilgisi or "Bilgi bulunamadı."
        }

    except Exception as e:
        hata_detayi = traceback.format_exc()
        print("🛑 HATA OLUŞTU:")
        print(hata_detayi)
        return JSONResponse(
            status_code=500,
            content={"hata": str(e), "detay": hata_detayi}
        )
