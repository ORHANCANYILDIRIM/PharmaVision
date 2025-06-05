import torch
from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path

def main():
    print("1")
    torch.cuda.empty_cache()  # Önbelleği temizle
    print(torch.cuda.memory_summary())  # Bellek kullanımını görüntüle
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

    # YOLOv8 modelini başlat
    # Model
    model = YOLO("yolov8n.pt")  # .to(device) gerekmez, YOLO otomatik atar

    # Eğitim (Windows için kritik optimizasyonlar)
    model.train(
        data="C:/Users/yldrm/Documents/GitHub/proje-kod-ve-raporu-ORHANCANYILDIRIM/Kutu Tespiti/Medicine-box-1/data.yaml",
        epochs=5,
        imgsz=640,
        batch=4,       # Batch boyutu küçültüldü
        workers=2,      # Windows'ta 2-4 arası optimal
        device=device,  # 0 veya 'cpu'
        single_cls=False,
        verbose=True    # Detaylı log
    )


        # Modeli değerlendirme (GÜNCEL YOLOv8.3.94 için)
    metrics = model.val()
    print("\nModel Değerlendirme Sonuçları:")
    print(f"mAP50: {metrics.box.map50:.4f}")  # 0.5 IoU'daki mAP
    print(f"mAP50-95: {metrics.box.map:.4f}") # 0.5-0.95 IoU'daki mAP
    print(f"Precision: {metrics.box.p[0]:.4f}")  # P (precision)
    print(f"Recall: {metrics.box.r[0]:.4f}")     # R (recall)
    print(f"F1 Score: {metrics.box.f1[0]:.4f}")  # F1 score

    # Modeli kaydetme (ONNX olmadan)
    model.save("medicine_model.pt")
    print("Model .pt formatında kaydedildi!")
    # Eğitim sonunda kayıt yolunu yazdır
    print(f"Model kaydedildi: {Path(model.trainer.save_dir).resolve()}")
    


    



if __name__ == '__main__':
    freeze_support()
    main()