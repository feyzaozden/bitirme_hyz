"""
class_mapper.py

Bu dosya, YOLO modelinden gelen sınıf isimlerini TEKNOFEST Havacılıkta Yapay Zeka
yarışmasında kullanılan sınıf ID'lerine dönüştürmek için hazırlanmıştır.

Şu an kullandığımız YOLO modelinde sınıflar:
- pedestrian
- car
- truck
- bus
- motor

TEKNOFEST şartnamesinde ise bu sınıflar daha genel tutulmuştur:
- Taşıt -> ID 0
- İnsan -> ID 1

Bu nedenle car, truck, bus ve motor sınıfları "Taşıt" olarak;
pedestrian sınıfı ise "İnsan" olarak eşlenmektedir.
"""


# YOLO modelinden gelen sınıf adlarını TEKNOFEST sınıf ID'lerine eşler.
YOLO_TO_TEKNOFEST_CLASS = {
    "car": "0",
    "truck": "0",
    "bus": "0",
    "motor": "0",
    "pedestrian": "1"
}


# TEKNOFEST sınıf ID'lerinin okunabilir karşılıkları.
TEKNOFEST_CLASS_NAMES = {
    "0": "Taşıt",
    "1": "İnsan",
    "2": "UAP",
    "3": "UAİ"
}


def map_yolo_class_to_teknofest(yolo_class_name: str) -> str | None:
    """
    YOLO sınıf adını TEKNOFEST sınıf ID'sine dönüştürür.

    Parametre:
        yolo_class_name (str): YOLO modelinden gelen sınıf adı.

    Döndürür:
        str | None:
            - "0" -> Taşıt
            - "1" -> İnsan
            - None -> Eşleşmeyen sınıf
    """

    return YOLO_TO_TEKNOFEST_CLASS.get(yolo_class_name)


def get_default_landing_status(teknofest_class_id: str) -> str:
    """
    TEKNOFEST formatında landing_status değerini üretir.

    Şu anki sistemimiz UAP/UAİ iniş alanı tespiti yapmadığı için
    taşıt ve insan sınıflarında landing_status değeri -1 olmalıdır.

    Döndürür:
        "-1" -> İniş alanı değil
    """

    return "-1"


def get_default_motion_status(teknofest_class_id: str) -> str:
    """
    TEKNOFEST formatında motion_status değerini üretir.

    Şimdilik basit başlangıç yaklaşımı:
    - Taşıt için: "0" yani hareketsiz varsayılır.
    - İnsan için: "-1" yani taşıt değil.

    İlerleyen aşamada ardışık kareler üzerinden hareketli/hareketsiz
    ayrımı yapılabilir.

    Parametre:
        teknofest_class_id (str): TEKNOFEST sınıf ID'si.

    Döndürür:
        str: motion_status değeri
    """

    if teknofest_class_id == "0":
        return "0"   # Taşıt, başlangıçta hareketsiz varsayılıyor.

    return "-1"      # Taşıt değil.