# Hava Aracı Görüntülerinde Çevresel Farkındalık İçin YOLO Tabanlı Nesne Tespit Sistemi

Bu proje, TEKNOFEST Havacılıkta Yapay Zekâ Yarışması kapsamında yer alan görevlerden hareketle geliştirilmiş bir bitirme çalışmasıdır. Projenin genel amacı, hava aracı görüntülerinden çevresel farkındalık sağlayan yapay zekâ tabanlı bir sistem geliştirmektir.

TEKNOFEST HYZ kapsamında sistem üç temel görev üzerinden ele alınmaktadır:

1. Nesne Tespiti
2. Pozisyon Kestirimi
3. Referans Obje / Görüntü Eşleme

Bu çalışmada öncelikli olarak birinci görev olan nesne tespiti modülü gerçekleştirilmiştir. Geliştirilen sistem, hava aracı görüntülerinde Taşıt, İnsan, UAP ve UAİ sınıflarını tespit etmekte; tespit sonuçlarını görsel çıktı ve JSON formatında üretmektedir.

---

## Proje Durumu

### Tamamlanan Modül: 1. Görev - Nesne Tespiti

Bu modülde TEKNOFEST HYZ 2025 Oturum 4 veri seti kullanılmıştır. XML formatındaki etiketler YOLO formatına dönüştürülmüş, veri seti dengeli şekilde eğitim, doğrulama ve test kümelerine ayrılmıştır.

Geliştirilen nesne tespit sistemi aşağıdaki sınıfları desteklemektedir:

| Sınıf ID | Sınıf |
|---:|---|
| 0 | Taşıt |
| 1 | İnsan |
| 2 | UAP |
| 3 | UAİ |

Model olarak YOLO11n kullanılmıştır. Eğitim sonucunda test veri kümesi üzerinde aşağıdaki sonuçlar elde edilmiştir:

| Metrik | Değer |
|---|---:|
| Precision | 0.977 |
| Recall | 0.956 |
| mAP@50 | 0.967 |
| mAP@50-95 | 0.797 |

---

## Pipeline Çıktıları

Geliştirilen pipeline verilen görüntü klasörünü otomatik olarak işler ve aşağıdaki çıktıları üretir:

- Kutulu tahmin görselleri
- Her görüntü için ayrı JSON dosyası
- Tüm görüntüler için toplu JSON dosyası
- Sınıf dağılımı ve durum bilgilerini içeren özet rapor

Örnek çıktı yapısı:

```text
outputs/demo_run
├── annotated_images
├── json_outputs
├── all_predictions.json
├── summary.json
└── summary.txt