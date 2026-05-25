import os
import cv2
import random
import numpy as np

# Ana veri yolu
BASE = r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\yolo_dataset"

# Eğitim görüntüleri ve etiketleri
IMAGE_DIR = os.path.join(BASE, "train", "images")
LABEL_DIR = os.path.join(BASE, "train", "labels")

# Sonuç klasörü
OUTPUT_DIR = r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\results\label_check"

# Sınıf isimleri
CLASS_NAMES = [
    "pedestrian",
    "car",
    "truck",
    "bus",
    "motor"
]

# Sonuç klasörü yoksa oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rastgele görüntü seç
image_files = os.listdir(IMAGE_DIR)
selected_image = random.choice(image_files)

# Görüntü ve etiket yolu
image_path = os.path.join(
    IMAGE_DIR,
    selected_image
)

label_path = os.path.join(
    LABEL_DIR,
    selected_image.replace(".jpg", ".txt")
)

# Türkçe karakter sorunu için güvenli görüntü okuma
image = cv2.imdecode(
    np.fromfile(
        image_path,
        dtype=np.uint8
    ),
    cv2.IMREAD_COLOR
)

# Görüntü boyutu
height, width, _ = image.shape

# Etiket dosyasını oku
with open(label_path, "r") as file:
    lines = file.readlines()

# Tüm kutuları çiz
for line in lines:

    class_id, x_center, y_center, box_width, box_height = map(
        float,
        line.split()
    )

    class_id = int(class_id)

    # Normalize edilmiş değerleri gerçek piksele çevir
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    # Sol üst ve sağ alt koordinatlar
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)

    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Kutu çiz
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Sınıf adını yaz
    cv2.putText(
        image,
        CLASS_NAMES[class_id],
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

# Kaydet
output_path = os.path.join(
    OUTPUT_DIR,
    selected_image
)

success, encoded_image = cv2.imencode(".jpg", image)

if success:
    encoded_image.tofile(output_path)
    print("Kontrol görüntüsü oluşturuldu:")
    print(output_path)
else:
    print("Görüntü kaydedilemedi.")


'''
Veri seti dönüşümünden sonra etiket doğrulama işlemi gerçekleştirilmiştir.
Bu aşamada Python ve OpenCV kütüphaneleri kullanılarak YOLO formatındaki etiketler 
tekrar piksel koordinatlarına dönüştürülmüş ve görüntü üzerine nesne kutuları çizdirilmiştir. 
Rastgele seçilen örnek görüntüler üzerinde yapılan incelemede nesne kutularının 
ilgili nesnelerle doğru şekilde eşleştiği gözlemlenmiştir. 
Böylece veri dönüşüm sürecinin doğru gerçekleştirildiği doğrulanmıştır.
'''