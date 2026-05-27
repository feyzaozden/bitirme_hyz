from pathlib import Path
import shutil
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from PIL import Image


# ============================================================
# TEKNOFEST HYZ 2025 - Dengeli XML -> YOLO Dönüşüm Script'i
# ============================================================
#
# Neden bu script yazıldı?
#
# İlk dönüşümde veri, video sırasına göre %70 train, %20 val,
# %10 test olarak bölünmüştü. Ancak video içindeki sınıflar zamana göre
# kümelendiği için bazı sınıflar bazı kümelerde hiç yer almadı.
#
# Örneğin:
# - Train içinde UAP yoktu.
# - Val içinde UAİ yoktu.
# - Test içinde İnsan yoktu.
#
# Bu durum model eğitimi ve değerlendirmesi için hatalıdır.
# Çünkü model, train içinde hiç görmediği bir sınıfı öğrenemez.
#
# Bu script, her görüntüdeki sınıfları dikkate alarak daha dengeli
# train / val / test dağılımı üretir.
# ============================================================


# ------------------------------------------------------------
# 1. Klasör yolları
# ------------------------------------------------------------

ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\raw\oturum_4"
)

OUTPUT_ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\yolo\oturum_4_balanced"
)

IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"


# ------------------------------------------------------------
# 2. Sınıf eşlemesi
# ------------------------------------------------------------
#
# Şartname sınıfları:
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
# 4. XML içinden sınıf ve bbox bilgilerini okuma
# ------------------------------------------------------------

def parse_xml_objects(xml_path: Path):
    """
    XML dosyasındaki nesneleri okur.

    Dönen liste örneği:
    [
        {
            "class_name": "Taşıt",
            "class_id": 0,
            "xmin": 12.0,
            "ymin": 40.0,
            "xmax": 80.0,
            "ymax": 100.0
        }
    ]
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []

    for obj in root.findall(".//object"):
        class_name = obj.findtext("name")

        if class_name not in CLASS_NAME_TO_ID:
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        objects.append(
            {
                "class_name": class_name,
                "class_id": CLASS_NAME_TO_ID[class_name],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return objects


# Her görüntü için nesne sınıf sayılarını çıkar.
stem_to_objects = {}
stem_to_class_counter = {}

total_class_counter = Counter()

for stem in common_stems:
    objects = parse_xml_objects(xml_map[stem])
    stem_to_objects[stem] = objects

    class_counter = Counter()

    for obj in objects:
        class_counter[obj["class_id"]] += 1
        total_class_counter[obj["class_id"]] += 1

    stem_to_class_counter[stem] = class_counter


print("\nGenel sınıf dağılımı:")
for class_id, class_name in YOLO_CLASS_NAMES.items():
    print(f"{class_id} - {class_name}: {total_class_counter[class_id]}")


# ------------------------------------------------------------
# 5. Dengeli train / val / test bölme
# ------------------------------------------------------------
#
# Mantık:
# - Her görüntüde hangi sınıflardan kaç tane var biliyoruz.
# - Her split için hedef görüntü sayısı belirliyoruz.
# - Ayrıca her split için hedef sınıf sayısı belirliyoruz.
# - Görüntüleri karıştırıp, sınıf eksikliği en fazla olan split'e atıyoruz.
#
# Bu yöntem tam mükemmel stratified split değildir ama bu proje için
# yeterince dengeli ve açıklanabilir bir çözümdür.
# ------------------------------------------------------------

random.seed(42)

all_stems = common_stems.copy()
random.shuffle(all_stems)

split_ratios = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

target_image_counts = {
    "train": int(len(all_stems) * split_ratios["train"]),
    "val": int(len(all_stems) * split_ratios["val"]),
}

target_image_counts["test"] = len(all_stems) - target_image_counts["train"] - target_image_counts["val"]

target_class_counts = {
    split_name: {
        class_id: total_class_counter[class_id] * ratio
        for class_id in YOLO_CLASS_NAMES.keys()
    }
    for split_name, ratio in split_ratios.items()
}

splits = {
    "train": [],
    "val": [],
    "test": [],
}

split_class_counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter(),
}


def split_score(split_name, stem):
    """
    Bir görüntüyü ilgili split'e koymanın ne kadar faydalı olduğunu hesaplar.

    Eğer split, o görüntüdeki sınıflara ihtiyaç duyuyorsa skor yüksek olur.
    """

    # Split görüntü sayısı hedefini geçtiyse bu split'e ekleme yapma.
    if len(splits[split_name]) >= target_image_counts[split_name]:
        return -10**9

    image_classes = stem_to_class_counter[stem]

    # Nesne içermeyen görüntüler için sadece kapasiteye göre davran.
    if not image_classes:
        return 0

    score = 0.0

    for class_id, count_in_image in image_classes.items():
        current_count = split_class_counts[split_name][class_id]
        target_count = target_class_counts[split_name][class_id]

        deficit = target_count - current_count

        if deficit > 0:
            score += min(deficit, count_in_image)

    # Daha küçük splitler boş kalmasın diye hafif kapasite bonusu
    remaining_capacity = target_image_counts[split_name] - len(splits[split_name])
    score += remaining_capacity * 0.0001

    return score


for stem in all_stems:
    scores = {
        split_name: split_score(split_name, stem)
        for split_name in ["train", "val", "test"]
    }

    best_split = max(scores, key=scores.get)

    splits[best_split].append(stem)
    split_class_counts[best_split].update(stem_to_class_counter[stem])


print("\nDengeli veri bölme sonucu:")
for split_name in ["train", "val", "test"]:
    print(f"\n{split_name.upper()}")
    print("Görüntü sayısı:", len(splits[split_name]))

    for class_id, class_name in YOLO_CLASS_NAMES.items():
        print(f"{class_id} - {class_name}: {split_class_counts[split_name][class_id]}")


# ------------------------------------------------------------
# 6. Çıktı klasörlerini temizle ve oluştur
# ------------------------------------------------------------

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

for split_name in ["train", "val", "test"]:
    (OUTPUT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / split_name / "labels").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 7. YOLO bbox dönüşümü
# ------------------------------------------------------------

def read_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size

    return width, height


def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    Pascal VOC formatını YOLO formatına dönüştürür.

    Pascal VOC:
        xmin, ymin, xmax, ymax

    YOLO:
        x_center, y_center, width, height

    YOLO değerleri 0-1 arasında normalize edilir.
    """

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


def objects_to_yolo_lines(objects, image_width, image_height):
    yolo_lines = []

    for obj in objects:
        yolo_bbox = convert_bbox_to_yolo(
            obj["xmin"],
            obj["ymin"],
            obj["xmax"],
            obj["ymax"],
            image_width,
            image_height,
        )

        if yolo_bbox is None:
            continue

        x_center, y_center, width, height = yolo_bbox

        yolo_line = (
            f"{obj['class_id']} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(yolo_line)

    return yolo_lines


# ------------------------------------------------------------
# 8. Görüntüleri ve etiketleri yaz
# ------------------------------------------------------------

for split_name, stems in splits.items():
    print(f"\n{split_name} yazılıyor...")

    for stem in stems:
        image_path = image_map[stem]
        output_image_path = OUTPUT_ROOT / split_name / "images" / image_path.name
        output_label_path = OUTPUT_ROOT / split_name / "labels" / f"{stem}.txt"

        shutil.copy2(image_path, output_image_path)

        image_width, image_height = read_image_size(image_path)

        yolo_lines = objects_to_yolo_lines(
            stem_to_objects[stem],
            image_width,
            image_height
        )

        output_label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )


# ------------------------------------------------------------
# 9. data.yaml oluştur
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
# 10. Sonuç özeti
# ------------------------------------------------------------

print("\n========== DENGELİ DÖNÜŞÜM TAMAMLANDI ==========")
print("YOLO veri seti yolu:")
print(OUTPUT_ROOT)

print("\ndata.yaml oluşturuldu:")
print(OUTPUT_ROOT / "data.yaml")