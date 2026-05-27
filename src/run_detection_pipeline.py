"""
run_detection_pipeline.py

Bu dosya, bitirme projesinin ana demo dosyasıdır.

Amaç:
Eğitilmiş YOLO11n modelini kullanarak hava aracı görüntülerinde nesne tespiti yapmak,
tespit edilen nesneleri görsel olarak kaydetmek ve her görüntü için TEKNOFEST benzeri
JSON çıktı üretmektir.

Sistem çıktıları:
1. Kutulu tahmin görselleri
2. Her görüntü için ayrı JSON dosyası
3. Tüm tahminleri içeren tek JSON dosyası
4. Özet rapor dosyası

Desteklenen sınıflar:
0 -> Taşıt
1 -> İnsan
2 -> UAP
3 -> UAİ

TEKNOFEST mantığı:
- Taşıt için motion_status kullanılır.
- İnsan için motion_status = -1 olur.
- UAP ve UAİ için landing_status kullanılır.
- Taşıt ve insan sınıfları iniş alanı olmadığı için landing_status = -1 olur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

from ultralytics import YOLO


CLASS_NAMES = {
    0: "Tasit",
    1: "Insan",
    2: "UAP",
    3: "UAI",
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def calculate_intersection_area(box_a: List[int], box_b: List[int]) -> float:
    """
    İki kutunun kesişim alanını hesaplar.

    Kutu formatı:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    return inter_width * inter_height


def calculate_box_area(box: List[int]) -> float:
    """
    Bir sınır kutusunun alanını hesaplar.
    """

    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def decide_landing_status(
    landing_box: List[int],
    obstacle_boxes: List[List[int]],
    threshold: float = 0.01,
) -> int:
    """
    UAP veya UAİ alanının inişe uygun olup olmadığını belirler.

    Mantık:
    - Eğer iniş alanı içinde taşıt veya insan varsa landing_status = 0 olur.
    - Eğer iniş alanı boşsa landing_status = 1 olur.

    threshold:
    İniş alanı ile engel kutusu arasındaki kesişim oranı bu değerden büyükse
    alan dolu kabul edilir.
    """

    landing_area = calculate_box_area(landing_box)

    if landing_area == 0:
        return 0

    for obstacle_box in obstacle_boxes:
        intersection_area = calculate_intersection_area(landing_box, obstacle_box)
        intersection_ratio = intersection_area / landing_area

        if intersection_ratio > threshold:
            return 0

    return 1


def get_box_center(box: List[int]) -> Tuple[float, float]:
    """
    Bir kutunun merkez noktasını döndürür.
    """

    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_center_distance(box_a: List[int], box_b: List[int]) -> float:
    """
    İki kutunun merkez noktaları arasındaki uzaklığı hesaplar.
    """

    ax, ay = get_box_center(box_a)
    bx, by = get_box_center(box_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def collect_image_files(source_path: Path) -> List[Path]:
    """
    Kaynak yol bir klasörse içindeki görüntüleri toplar.
    Kaynak yol tek bir görüntüyse onu liste olarak döndürür.
    """

    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Desteklenmeyen dosya türü: {source_path}")
        return [source_path]

    if source_path.is_dir():
        image_files = [
            file
            for file in source_path.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(image_files)

    raise FileNotFoundError(f"Kaynak yol bulunamadı: {source_path}")


def convert_result_to_raw_detections(result) -> List[Dict[str, Any]]:
    """
    Ultralytics YOLO sonucunu sade bir liste formatına dönüştürür.
    """

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bbox = [
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        ]

        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
        )

    return detections


def assign_motion_status(
    all_frame_detections: List[Dict[str, Any]],
    movement_threshold: float = 8.0,
    matching_threshold: float = 50.0,
) -> None:
    """
    Taşıt sınıfı için basit hareket durumu ataması yapar.

    Bu yöntem ardışık karelerdeki taşıt kutularının merkez noktalarını karşılaştırır.

    motion_status:
    - 0 -> Hareketsiz
    - 1 -> Hareketli
    - -1 -> Taşıt değil

    Not:
    Hava aracı kamerası da hareket ettiği için bu yöntem yarışma düzeyinde kusursuz
    bir hareket analizi değildir. Bitirme çalışmasında temel bir prototip yaklaşımı
    olarak kullanılır.
    """

    previous_vehicle_boxes: List[List[int]] = []

    for frame_data in all_frame_detections:
        current_vehicle_boxes = [
            detection["bbox"]
            for detection in frame_data["detections_raw"]
            if detection["class_id"] == 0
        ]

        for detection in frame_data["detections_raw"]:
            if detection["class_id"] != 0:
                detection["motion_status"] = -1
                continue

            if not previous_vehicle_boxes:
                detection["motion_status"] = 0
                continue

            distances = [
                calculate_center_distance(detection["bbox"], previous_box)
                for previous_box in previous_vehicle_boxes
            ]

            nearest_distance = min(distances)

            if nearest_distance > matching_threshold:
                detection["motion_status"] = 0
            elif nearest_distance > movement_threshold:
                detection["motion_status"] = 1
            else:
                detection["motion_status"] = 0

        previous_vehicle_boxes = current_vehicle_boxes


def build_teknofest_json(frame_name: str, detections_raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ham YOLO tahminlerini TEKNOFEST benzeri JSON formatına dönüştürür.
    """

    obstacle_boxes = [
        detection["bbox"]
        for detection in detections_raw
        if detection["class_id"] in [0, 1]
    ]

    detections_json = []

    for detection in detections_raw:
        class_id = detection["class_id"]
        x1, y1, x2, y2 = detection["bbox"]

        landing_status = -1
        motion_status = detection.get("motion_status", -1)

        if class_id in [2, 3]:
            landing_status = decide_landing_status(
                landing_box=detection["bbox"],
                obstacle_boxes=obstacle_boxes,
            )

        if class_id == 0 and motion_status not in [0, 1]:
            motion_status = 0

        detection_json = {
            "cls": class_id,
            "landing_status": landing_status,
            "motion_status": motion_status,
            "top_left_x": x1,
            "top_left_y": y1,
            "bottom_right_x": x2,
            "bottom_right_y": y2,
        }

        detections_json.append(detection_json)

    return {
        "frame": frame_name,
        "detections": detections_json,
    }


def save_summary(
    output_dir: Path,
    all_json_results: List[Dict[str, Any]],
) -> None:
    """
    Pipeline sonunda genel özet dosyaları oluşturur.
    """

    class_counter = Counter()
    landing_status_counter = Counter()
    motion_status_counter = Counter()
    total_detection_count = 0

    for frame_result in all_json_results:
        detections = frame_result["detections"]
        total_detection_count += len(detections)

        for detection in detections:
            class_counter[detection["cls"]] += 1
            landing_status_counter[detection["landing_status"]] += 1
            motion_status_counter[detection["motion_status"]] += 1

    summary = {
        "total_frames": len(all_json_results),
        "total_detections": total_detection_count,
        "class_distribution": {
            "0_Tasit": class_counter[0],
            "1_Insan": class_counter[1],
            "2_UAP": class_counter[2],
            "3_UAI": class_counter[3],
        },
        "landing_status_distribution": dict(landing_status_counter),
        "motion_status_distribution": dict(motion_status_counter),
    }

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    summary_text = (
        "TEKNOFEST Nesne Tespit Pipeline Özeti\n"
        "=====================================\n\n"
        f"İşlenen görüntü sayısı: {summary['total_frames']}\n"
        f"Toplam tespit sayısı: {summary['total_detections']}\n\n"
        "Sınıf dağılımı:\n"
        f"  Taşıt: {summary['class_distribution']['0_Tasit']}\n"
        f"  İnsan: {summary['class_distribution']['1_Insan']}\n"
        f"  UAP: {summary['class_distribution']['2_UAP']}\n"
        f"  UAİ: {summary['class_distribution']['3_UAI']}\n\n"
        "Landing status dağılımı:\n"
        f"  {summary['landing_status_distribution']}\n\n"
        "Motion status dağılımı:\n"
        f"  {summary['motion_status_distribution']}\n"
    )

    summary_txt_path.write_text(summary_text, encoding="utf-8")


def run_pipeline(
    model_path: Path,
    source_path: Path,
    output_dir: Path,
    confidence_threshold: float,
) -> None:
    """
    Ana pipeline fonksiyonu.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated_images"
    json_dir = output_dir / "json_outputs"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_image_files(source_path)

    if not image_files:
        raise RuntimeError("İşlenecek görüntü bulunamadı.")

    model = YOLO(str(model_path))

    all_frame_detections = []

    print("Model yüklendi.")
    print("İşlenecek görüntü sayısı:", len(image_files))

    for image_path in image_files:
        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False,
        )

        result = results[0]

        # Kutulu görseli kaydet
        annotated_output_path = annotated_dir / f"{image_path.stem}.jpg"
        result.save(filename=str(annotated_output_path))

        detections_raw = convert_result_to_raw_detections(result)

        all_frame_detections.append(
            {
                "frame": image_path.name,
                "detections_raw": detections_raw,
            }
        )

    # Taşıt hareket durumunu basit yöntemle ata
    assign_motion_status(all_frame_detections)

    all_json_results = []

    for frame_data in all_frame_detections:
        frame_json = build_teknofest_json(
            frame_name=frame_data["frame"],
            detections_raw=frame_data["detections_raw"],
        )

        all_json_results.append(frame_json)

        json_output_path = json_dir / f"{Path(frame_data['frame']).stem}.json"

        json_output_path.write_text(
            json.dumps(frame_json, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

    all_predictions_path = output_dir / "all_predictions.json"
    all_predictions_path.write_text(
        json.dumps(all_json_results, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    save_summary(output_dir, all_json_results)

    print("\nPipeline tamamlandı.")
    print("Çıktı klasörü:", output_dir)
    print("Kutulu görseller:", annotated_dir)
    print("JSON çıktıları:", json_dir)
    print("Toplu JSON:", all_predictions_path)
    print("Özet dosyası:", output_dir / "summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO modeli ile TEKNOFEST formatında nesne tespit pipeline'ı"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Eğitilmiş YOLO model dosyası yolu. Örnek: models/best.pt",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Görüntü klasörü veya tek görüntü yolu.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/demo_run",
        help="Çıktıların kaydedileceği klasör.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Güven eşiği. Varsayılan: 0.25",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        model_path=Path(args.model),
        source_path=Path(args.source),
        output_dir=Path(args.output),
        confidence_threshold=args.conf,
    )