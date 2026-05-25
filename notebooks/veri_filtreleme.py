import os
import random
import shutil
from PIL import Image

BASE = r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset"

RAW_TRAIN = os.path.join(BASE, "raw_train")
RAW_VAL = os.path.join(BASE, "raw_val")
YOLO_DATASET = os.path.join(BASE, "yolo_dataset")

SELECTED_CLASSES = {
    1: 0,   # pedestrian
    4: 1,   # car
    6: 2,   # truck
    9: 3,   # bus
    10: 4   # motor
}

CLASS_NAMES = ["pedestrian", "car", "truck", "bus", "motor"]

random.seed(42)


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


for split in ["train", "val", "test"]:
    clear_folder(os.path.join(YOLO_DATASET, split, "images"))
    clear_folder(os.path.join(YOLO_DATASET, split, "labels"))


def convert_annotation(annotation_path, image_path, output_label_path):
    image = Image.open(image_path)
    img_w, img_h = image.size

    yolo_lines = []

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split(",")

        if len(values) < 8:
            continue

        x = float(values[0])
        y = float(values[1])
        w = float(values[2])
        h = float(values[3])
        score = int(values[4])
        class_id = int(values[5])

        if score == 0:
            continue

        if class_id not in SELECTED_CLASSES:
            continue

        new_class = SELECTED_CLASSES[class_id]

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        yolo_lines.append(
            f"{new_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    if len(yolo_lines) == 0:
        return False

    with open(output_label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    return True


def create_split(source_folder, split_name, count):
    image_dir = os.path.join(source_folder, "images")
    annotation_dir = os.path.join(source_folder, "annotations")

    image_files = list(os.listdir(image_dir))
    random.shuffle(image_files)

    selected = 0

    for image_file in image_files:
        if selected >= count:
            break

        image_path = os.path.join(image_dir, image_file)
        annotation_file = image_file.replace(".jpg", ".txt")
        annotation_path = os.path.join(annotation_dir, annotation_file)

        if not os.path.exists(annotation_path):
            continue

        output_image_path = os.path.join(
            YOLO_DATASET, split_name, "images", image_file
        )

        output_label_path = os.path.join(
            YOLO_DATASET, split_name, "labels", annotation_file
        )

        success = convert_annotation(
            annotation_path,
            image_path,
            output_label_path
        )

        if success:
            shutil.copy(image_path, output_image_path)
            selected += 1

    print(f"{split_name} için seçilen görüntü sayısı: {selected}")


create_split(RAW_TRAIN, "train", 300)
create_split(RAW_VAL, "val", 80)
create_split(RAW_VAL, "test", 40)


yaml_path = os.path.join(YOLO_DATASET, "data.yaml")

with open(yaml_path, "w") as f:
    f.write(f"""path: {YOLO_DATASET}
train: train/images
val: val/images
test: test/images

names:
  0: pedestrian
  1: car
  2: truck
  3: bus
  4: motor
""")

print("YOLO veri seti oluşturuldu.")
print("data.yaml oluşturuldu.")



'''
Bu aşamada VisDrone2019-DET veri seti, YOLO formatına dönüştürülmüştür. 
Orijinal VisDrone etiketleri x, y, width, height, score, class_id, truncation ve occlusion 
bilgilerini içermektedir. YOLO algoritması ise her nesne için class_id, x_center, 
y_center, width ve height değerlerini normalize edilmiş biçimde istemektedir. 

Bu nedenle Python ile bir dönüştürme betiği hazırlanmıştır. Betik öncelikle ham 
eğitim ve doğrulama klasörlerinden görüntü ve etiket dosyalarını okumuştur. Ardından 
yalnızca proje kapsamına alınan pedestrian, car, truck, bus ve motor sınıfları filtrelenmiştir. 
Seçilen sınıfların VisDrone sınıf numaraları YOLO için 0’dan başlayacak şekilde yeniden 
numaralandırılmıştır. Son aşamada x, y, width ve height değerleri görüntü genişliği ve 
yüksekliğine bölünerek normalize edilmiş; böylece YOLO eğitiminde kullanılabilecek 
etiket dosyaları oluşturulmuştur.

Hazırlanan veri seti 300 eğitim, 80 doğrulama ve 40 test görüntüsünden oluşmaktadır. 
Bu yapı, modelin eğitim sırasında öğrenmesini, doğrulama sırasında performansının 
izlenmesini ve test aşamasında daha önce görmediği görüntüler üzerinde değerlendirilmesini 
sağlamaktadır.
'''
