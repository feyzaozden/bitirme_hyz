from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


# ============================================================
# TEKNOFEST HYZ 2025 - Oturum 4 Veri İnceleme Script'i
# ============================================================
#
# Amaç:
# Bu script, indirdiğimiz TEKNOFEST Oturum 4 verisinin yapısını anlamak için yazılmıştır.
#
# Kontrol edilenler:
# 1. Kaç görüntü dosyası var?
# 2. Kaç XML etiket dosyası var?
# 3. Görüntü ve etiket isimleri eşleşiyor mu?
# 4. XML dosyalarının içinde hangi sınıflar var?
# 5. İlk dolu XML dosyası nasıl görünüyor?
#
# Bu aşamada eğitim yapmıyoruz.
# Sadece veri setini tanıyoruz.
# ============================================================


# Ana veri yolu
ROOT = Path(
    r"C:\Users\fyzoz\OneDrive\Masaüstü\Bitirme_Projesi\dataset\teknofest\raw\oturum_4"
)

# Görüntü ve etiket dosyalarını alt klasörlerden otomatik bul
image_files = sorted((ROOT / "images").rglob("*.webp"))
xml_files = sorted((ROOT / "labels").rglob("*.xml"))

print("========== DOSYA SAYILARI ==========")
print("Görüntü sayısı:", len(image_files))
print("XML etiket sayısı:", len(xml_files))


# Dosya isimleri eşleşiyor mu kontrol et
image_stems = {file.stem for file in image_files}
xml_stems = {file.stem for file in xml_files}

matched = image_stems.intersection(xml_stems)
missing_xml = image_stems - xml_stems
missing_image = xml_stems - image_stems

print("\n========== EŞLEŞME KONTROLÜ ==========")
print("Eşleşen görüntü-etiket sayısı:", len(matched))
print("Etiketi olmayan görüntü sayısı:", len(missing_xml))
print("Görüntüsü olmayan etiket sayısı:", len(missing_image))


# XML içinden Pascal VOC benzeri object/name/bndbox yapısını okumaya çalış
def parse_objects_from_xml(xml_path):
    """
    XML dosyasındaki nesneleri okumaya çalışır.

    Beklenen olası yapı:
    <object>
        <name>...</name>
        <bndbox>
            <xmin>...</xmin>
            <ymin>...</ymin>
            <xmax>...</xmax>
            <ymax>...</ymax>
        </bndbox>
    </object>

    Eğer XML farklı formattaysa, bu fonksiyon boş liste döndürebilir.
    O durumda XML'in ham içeriğine bakacağız.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []

    for obj in root.findall(".//object"):
        name = obj.findtext("name")

        bbox = obj.find("bndbox")

        if bbox is not None:
            xmin = bbox.findtext("xmin")
            ymin = bbox.findtext("ymin")
            xmax = bbox.findtext("xmax")
            ymax = bbox.findtext("ymax")
        else:
            xmin = ymin = xmax = ymax = None

        objects.append(
            {
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return objects


class_counter = Counter()
non_empty_examples = []

for xml_file in xml_files:
    try:
        objects = parse_objects_from_xml(xml_file)

        if objects:
            non_empty_examples.append((xml_file, objects))

            for obj in objects:
                class_counter[obj["name"]] += 1

    except Exception as error:
        print("XML okunurken hata oluştu:", xml_file)
        print(error)


print("\n========== SINIF DAĞILIMI ==========")

if class_counter:
    for class_name, count in class_counter.items():
        print(class_name, ":", count)
else:
    print("Pascal VOC tarzında object/name bilgisi bulunamadı.")
    print("XML yapısı farklı olabilir. İlk XML içeriğini aşağıda yazdırıyorum.")


print("\n========== İLK 3 GÖRÜNTÜ ==========")
for file in image_files[:3]:
    print(file)


print("\n========== İLK 3 XML ==========")
for file in xml_files[:3]:
    print(file)


print("\n========== İLK DOLU XML ÖRNEĞİ ==========")

if non_empty_examples:
    xml_file, objects = non_empty_examples[0]

    print("Dosya:", xml_file)
    print("Nesneler:")

    for obj in objects:
        print(obj)

else:
    # Eğer object/name yapısı bulunamadıysa ilk XML dosyasının ham içeriğini göster
    if xml_files:
        first_xml = xml_files[0]

        print("İlk XML dosyası:", first_xml)
        print("\nXML içeriği:")
        print(first_xml.read_text(encoding="utf-8", errors="ignore"))