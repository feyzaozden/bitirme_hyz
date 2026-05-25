from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# TEKNOFEST YOLO Etiket Kontrol Script'i
# ============================================================
#
# Amaç:
# XML -> YOLO dönüşümünden sonra oluşan etiketlerin doğru yerde
# kutu çizip çizmediğini görsel olarak kontrol etmek.
#
# Bu aşamada model eğitimi yapılmaz.
# Sadece etiketlerin doğru dönüştürülüp dönüştürülmediği incelenir.
#
# Kontrol edilen şey:
# - class_id doğru mu?
# - x_center, y_center, width, height doğru şekilde piksele çevriliyor mu?
# - kutular nesnelerin üstüne oturuyor mu?
# ============================================================


DATASET_ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\yolo\oturum_4"
)

OUTPUT_DIR = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\results\teknofest_label_check"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


# Her sınıf için farklı renk kullanıyoruz.
CLASS_COLORS = {
    0: "cyan",
    1: "lime",
    2: "yellow",
    3: "red",
}


def yolo_to_pixel_bbox(line, image_width, image_height):
    """
    YOLO formatındaki bir etiketi piksel koordinatlarına çevirir.

    YOLO formatı:
        class_id x_center y_center width height

    Bu değerler 0-1 arasında normalize edilmiştir.
    Görüntü üzerine kutu çizebilmek için tekrar piksel değerine çevrilir.
    """

    parts = line.strip().split()

    if len(parts) != 5:
        return None

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    box_width = float(parts[3])
    box_height = float(parts[4])

    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return class_id, x1, y1, x2, y2


def draw_labels_on_image(image_path, label_path, output_path):
    """
    Bir görüntü üzerine YOLO etiketlerini kutu olarak çizer.
    """

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    label_text = label_path.read_text(encoding="utf-8").strip()

    if not label_text:
        return False

    lines = label_text.splitlines()

    for line in lines:
        bbox = yolo_to_pixel_bbox(line, image_width, image_height)

        if bbox is None:
            continue

        class_id, x1, y1, x2, y2 = bbox

        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = CLASS_COLORS.get(class_id, "white")

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2
        )

        draw.text(
            (x1, max(0, y1 - 12)),
            class_name,
            fill=color
        )

    image.save(output_path)
    return True


def collect_non_empty_samples(split_name, sample_count=10):
    """
    Belirli bir split içinden etiketi boş olmayan örnekler seçer.
    """

    images_dir = DATASET_ROOT / split_name / "images"
    labels_dir = DATASET_ROOT / split_name / "labels"

    label_files = sorted(labels_dir.glob("*.txt"))

    non_empty_pairs = []

    for label_path in label_files:
        if label_path.read_text(encoding="utf-8").strip():
            image_path = images_dir / f"{label_path.stem}.webp"

            if image_path.exists():
                non_empty_pairs.append((image_path, label_path))

    random.seed(42)
    random.shuffle(non_empty_pairs)

    return non_empty_pairs[:sample_count]


# Train, val ve test içinden örnek kontrol görüntüleri oluştur.
total_saved = 0

for split_name in ["train", "val", "test"]:
    samples = collect_non_empty_samples(split_name, sample_count=8)

    split_output_dir = OUTPUT_DIR / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{split_name} için seçilen örnek sayısı:", len(samples))

    for image_path, label_path in samples:
        output_path = split_output_dir / f"{image_path.stem}_check.jpg"

        success = draw_labels_on_image(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path
        )

        if success:
            total_saved += 1


print("\nKontrol görüntüleri oluşturuldu.")
print("Toplam çıktı sayısı:", total_saved)
print("Çıktı klasörü:")
print(OUTPUT_DIR)