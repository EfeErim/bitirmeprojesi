# Validation Dosyaları

Bu bölüm validation split artefactlerini toplar.

## Validation gate özeti
- Yol: `validation/metric_gate.json`
- Öncelik: `high`
- Format: `json`
- Amaç: Test öncesi referans performansı görmek
- Açıklama: Validation split üzerinde ölçülen yardımcı gate kararı.

## Validation OOD kanıt özeti
- Yol: `validation/ood_evidence_summary.json`
- Öncelik: `high`
- Format: `json`
- Amaç: OOD kanıtının yeterliliğini görmek
- Açıklama: Bu split için OOD örnek sayıları ve özet metrikleri gösterir.

## Validation sınıflandırma raporu
- Yol: `validation/classification_report.txt`
- Öncelik: `medium`
- Format: `txt`
- Amaç: Sınıf bazlı metrikleri metin olarak okumak
- Açıklama: Precision, recall ve F1 özetini metin formatında sunar.

## Validation sınıflandırma raporu JSON
- Yol: `validation/classification_report.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Raporu programatik olarak tüketmek
- Açıklama: Aynı raporun makine-okur JSON sürümü.

## Validation sınıf bazlı metrikler
- Yol: `validation/per_class_metrics.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Hangi sınıfta sorun olduğunu görmek
- Açıklama: Her sınıf için precision, recall, F1 ve support değerlerini içerir.

## Validation confusion matrix CSV
- Yol: `validation/confusion_matrix.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Karışan sınıf çiftlerini sayısal görmek
- Açıklama: Ham confusion matrix değerlerini tablo halinde tutar.

## Validation confusion matrix görseli
- Yol: `validation/confusion_matrix.png`
- Öncelik: `medium`
- Format: `png`
- Amaç: Hangi sınıfların karıştığını hızlı görmek
- Açıklama: Ham confusion matrix'in görsel hali.

## Validation normalize confusion matrix
- Yol: `validation/confusion_matrix_normalized.png`
- Öncelik: `medium`
- Format: `png`
- Amaç: Oransal hata desenini görmek
- Açıklama: Sınıf büyüklüğünden bağımsız normalize confusion matrix görseli.
