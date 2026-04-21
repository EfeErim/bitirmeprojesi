# Test Dosyaları

Bu bölüm held-out test artefactlerini toplar.

## Test gate özeti
- Yol: `test/metric_gate.json`
- Öncelik: `critical`
- Format: `json`
- Amaç: Asıl test kararını görmek
- Açıklama: Held-out test performansının gate kararını içerir.

## Test OOD kanıt özeti
- Yol: `test/ood_evidence_summary.json`
- Öncelik: `high`
- Format: `json`
- Amaç: OOD kanıtının yeterliliğini görmek
- Açıklama: Bu split için OOD örnek sayıları ve özet metrikleri gösterir.

## Test sınıflandırma raporu
- Yol: `test/classification_report.txt`
- Öncelik: `medium`
- Format: `txt`
- Amaç: Sınıf bazlı metrikleri metin olarak okumak
- Açıklama: Precision, recall ve F1 özetini metin formatında sunar.

## Test sınıflandırma raporu JSON
- Yol: `test/classification_report.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Raporu programatik olarak tüketmek
- Açıklama: Aynı raporun makine-okur JSON sürümü.

## Test sınıf bazlı metrikler
- Yol: `test/per_class_metrics.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Hangi sınıfta sorun olduğunu görmek
- Açıklama: Her sınıf için precision, recall, F1 ve support değerlerini içerir.

## Test confusion matrix CSV
- Yol: `test/confusion_matrix.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Karışan sınıf çiftlerini sayısal görmek
- Açıklama: Ham confusion matrix değerlerini tablo halinde tutar.

## Test confusion matrix görseli
- Yol: `test/confusion_matrix.png`
- Öncelik: `medium`
- Format: `png`
- Amaç: Hangi sınıfların karıştığını hızlı görmek
- Açıklama: Ham confusion matrix'in görsel hali.

## Test normalize confusion matrix
- Yol: `test/confusion_matrix_normalized.png`
- Öncelik: `medium`
- Format: `png`
- Amaç: Oransal hata desenini görmek
- Açıklama: Sınıf büyüklüğünden bağımsız normalize confusion matrix görseli.
