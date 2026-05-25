from pathlib import Path
from collections import Counter


# ============================================================
# TEKNOFEST YOLO Veri Seti - Split Bazlı Sınıf Analizi
# ============================================================
#
# Amaç:
# XML -> YOLO dönüşümünden sonra train, val ve test klasörlerinde
# hangi sınıftan kaç nesne olduğunu görmek.
#
# Bu analiz raporda da kullanılabilir.
# Çünkü veri setinin nasıl dağıtıldığını sayısal olarak gösterir.
# ============================================================


DATASET_ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\yolo\oturum_4"
)

CLASS_NAMES = {
    "0": "Tasit",
    "1": "Insan",
    "2": "UAP",
    "3": "UAI",
}


def count_classes_in_split(split_name):
    labels_dir = DATASET_ROOT / split_name / "labels"

    counter = Counter()
    image_with_object_count = 0
    empty_label_count = 0

    label_files = sorted(labels_dir.glob("*.txt"))

    for label_file in label_files:
        content = label_file.read_text(encoding="utf-8").strip()

        if not content:
            empty_label_count += 1
            continue

        image_with_object_count += 1

        for line in content.splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = parts[0]
            counter[class_id] += 1

    return {
        "total_label_files": len(label_files),
        "image_with_object_count": image_with_object_count,
        "empty_label_count": empty_label_count,
        "class_counter": counter,
    }


general_counter = Counter()

for split_name in ["train", "val", "test"]:
    result = count_classes_in_split(split_name)

    print("\n==========", split_name.upper(), "==========")
    print("Etiket dosyası sayısı:", result["total_label_files"])
    print("Nesne içeren görüntü sayısı:", result["image_with_object_count"])
    print("Boş etiket dosyası sayısı:", result["empty_label_count"])

    print("\nSınıf dağılımı:")

    for class_id, class_name in CLASS_NAMES.items():
        count = result["class_counter"].get(class_id, 0)
        print(f"{class_id} - {class_name}: {count}")

    general_counter.update(result["class_counter"])


print("\n========== GENEL TOPLAM ==========")

for class_id, class_name in CLASS_NAMES.items():
    count = general_counter.get(class_id, 0)
    print(f"{class_id} - {class_name}: {count}")