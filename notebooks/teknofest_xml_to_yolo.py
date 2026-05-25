from pathlib import Path
import shutil
import random
import xml.etree.ElementTree as ET
from collections import Counter

from PIL import Image


# ============================================================
# TEKNOFEST HYZ 2025 - XML -> YOLO Dönüşüm Script'i
# ============================================================
#
# Amaç:
# TEKNOFEST tarafından verilen Pascal VOC benzeri XML etiketlerini
# YOLO formatına dönüştürmek.
#
# Girdi:
# - .webp görüntüler
# - .xml etiket dosyaları
#
# Çıktı:
# dataset/teknofest/yolo/oturum_4 altında:
#
# train/images
# train/labels
# val/images
# val/labels
# test/images
# test/labels
# data.yaml
#
# Neden gerekli?
# YOLO modelleri eğitim sırasında etiketleri şu formatta ister:
#
# class_id x_center y_center width height
#
# Bu değerler piksel değil, 0-1 arasında normalize edilmiş olmalıdır.
# ============================================================


# ------------------------------------------------------------
# 1. Ana klasör yolları
# ------------------------------------------------------------

ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\raw\oturum_4"
)

OUTPUT_ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\yolo\oturum_4"
)

IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"


# ------------------------------------------------------------
# 2. TEKNOFEST sınıf eşlemesi
# ------------------------------------------------------------
#
# XML dosyalarında sınıf isimleri Türkçe olarak geliyor:
# - Taşıt
# - İnsan
# - UAP
# - UAİ
#
# YOLO için bunları sayısal ID'ye çeviriyoruz.
#
# Şartnameye uygun sınıf yapısı:
# 0 -> Taşıt
# 1 -> İnsan
# 2 -> UAP
# 3 -> UAİ
# ------------------------------------------------------------

CLASS_NAME_TO_ID = {
    "Taşıt": 0,
    "İnsan": 1,
    "UAP": 2,
    "UAİ": 3,
}


# data.yaml içinde Türkçe karakter sorunlarını azaltmak için
# sınıf adlarını ASCII karakterlerle yazıyoruz.
YOLO_CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


# ------------------------------------------------------------
# 3. Dosyaları bul
# ------------------------------------------------------------

image_files = sorted(IMAGES_DIR.rglob("*.webp"))
xml_files = sorted(LABELS_DIR.rglob("*.xml"))

image_map = {image_file.stem: image_file for image_file in image_files}
xml_map = {xml_file.stem: xml_file for xml_file in xml_files}

common_stems = sorted(set(image_map.keys()).intersection(set(xml_map.keys())))

print("Görüntü sayısı:", len(image_files))
print("XML sayısı:", len(xml_files))
print("Eşleşen dosya sayısı:", len(common_stems))


if not common_stems:
    raise RuntimeError("Eşleşen görüntü ve XML dosyası bulunamadı.")


# ------------------------------------------------------------
# 4. Zaman sıralı train/val/test bölme
# ------------------------------------------------------------
#
# Bu veri bir video oturumundan geldiği için kareler ardışık.
# Rastgele bölmek yerine zamana göre bölmek daha mantıklı.
#
# Neden?
# Çünkü art arda gelen kareler birbirine çok benzer.
# Rastgele bölersek train ve test içinde neredeyse aynı kareler olabilir.
# Bu da sonuçları olduğundan iyi gösterebilir.
#
# Bölme oranı:
# %70 train
# %20 validation
# %10 test
# ------------------------------------------------------------

total_count = len(common_stems)

train_end = int(total_count * 0.70)
val_end = int(total_count * 0.90)

splits = {
    "train": common_stems[:train_end],
    "val": common_stems[train_end:val_end],
    "test": common_stems[val_end:],
}

print("\nVeri bölme:")
print("Train:", len(splits["train"]))
print("Val:", len(splits["val"]))
print("Test:", len(splits["test"]))


# ------------------------------------------------------------
# 5. Çıktı klasörlerini hazırla
# ------------------------------------------------------------

for split_name in ["train", "val", "test"]:
    (OUTPUT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / split_name / "labels").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 6. XML okuma fonksiyonu
# ------------------------------------------------------------

def read_image_size(image_path: Path):
    """
    Görüntünün genişlik ve yükseklik bilgisini okur.

    .webp dosyalarını güvenli okumak için Pillow kullanıyoruz.
    """

    with Image.open(image_path) as img:
        width, height = img.size

    return width, height


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    Pascal VOC bbox formatını YOLO formatına dönüştürür.

    Pascal VOC:
        xmin, ymin, xmax, ymax

    YOLO:
        x_center, y_center, width, height

    Tüm YOLO değerleri görüntü genişliği/yüksekliğine bölünerek
    normalize edilir.
    """

    # Sınır dışına taşan kutuları görüntü sınırlarına kırpıyoruz.
    xmin = max(0.0, min(float(xmin), image_width))
    ymin = max(0.0, min(float(ymin), image_height))
    xmax = max(0.0, min(float(xmax), image_width))
    ymax = max(0.0, min(float(ymax), image_height))

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        return None

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height


def parse_xml_to_yolo_lines(xml_path: Path, image_width: int, image_height: int):
    """
    Bir XML dosyasındaki nesneleri YOLO satırlarına dönüştürür.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_lines = []
    class_counter = Counter()

    for obj in root.findall(".//object"):
        class_name = obj.findtext("name")

        if class_name not in CLASS_NAME_TO_ID:
            print(f"Uyarı: Tanınmayan sınıf atlandı: {class_name}")
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = bbox.findtext("xmin")
        ymin = bbox.findtext("ymin")
        xmax = bbox.findtext("xmax")
        ymax = bbox.findtext("ymax")

        yolo_bbox = convert_bbox_to_yolo(
            xmin, ymin, xmax, ymax, image_width, image_height
        )

        if yolo_bbox is None:
            continue

        class_id = CLASS_NAME_TO_ID[class_name]

        x_center, y_center, width, height = yolo_bbox

        yolo_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(yolo_line)
        class_counter[class_name] += 1

    return yolo_lines, class_counter


# ------------------------------------------------------------
# 7. Dönüştürme işlemi
# ------------------------------------------------------------

total_class_counter = Counter()
split_object_counter = Counter()

for split_name, stems in splits.items():
    print(f"\n{split_name} dönüştürülüyor...")

    for stem in stems:
        image_path = image_map[stem]
        xml_path = xml_map[stem]

        output_image_path = OUTPUT_ROOT / split_name / "images" / image_path.name
        output_label_path = OUTPUT_ROOT / split_name / "labels" / f"{stem}.txt"

        # Görüntüyü yeni YOLO veri seti klasörüne kopyala
        shutil.copy2(image_path, output_image_path)

        image_width, image_height = read_image_size(image_path)

        yolo_lines, class_counter = parse_xml_to_yolo_lines(
            xml_path,
            image_width,
            image_height
        )

        # Etiket dosyasını yaz.
        # Nesne olmasa bile boş txt dosyası oluşturulur.
        output_label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )

        total_class_counter.update(class_counter)
        split_object_counter[split_name] += len(yolo_lines)


# ------------------------------------------------------------
# 8. data.yaml oluştur
# ------------------------------------------------------------

data_yaml = f"""path: {OUTPUT_ROOT.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: {YOLO_CLASS_NAMES[0]}
  1: {YOLO_CLASS_NAMES[1]}
  2: {YOLO_CLASS_NAMES[2]}
  3: {YOLO_CLASS_NAMES[3]}
"""

(OUTPUT_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")


# ------------------------------------------------------------
# 9. Sonuç özeti
# ------------------------------------------------------------

print("\n========== DÖNÜŞÜM TAMAMLANDI ==========")
print("YOLO veri seti yolu:")
print(OUTPUT_ROOT)

print("\nSınıf dağılımı:")
for class_name, count in total_class_counter.items():
    print(class_name, ":", count)

print("\nSplit bazında nesne sayısı:")
for split_name, count in split_object_counter.items():
    print(split_name, ":", count)

print("\ndata.yaml oluşturuldu:")
print(OUTPUT_ROOT / "data.yaml")