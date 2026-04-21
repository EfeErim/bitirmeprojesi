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

## Test hard examples
- Yol: `test/hard_examples.csv`
- Öncelik: `high`
- Format: `csv`
- Amaç: Feedback ile düzeltilecek zor örnekleri önceliklendirmek
- Açıklama: Yanlış sınıflanan veya kaçırılan OOD örneklerini öncelikli inceleme için listeler.

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

## Test OOD tip kırılımı
- Yol: `test/ood_type_breakdown.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Hangi OOD alt tipinde sorun olduğunu görmek
- Açıklama: OOD klasör alt tiplerine göre metrik kırılımını içerir.

## Test OOD method comparison
- Yol: `test/ood_method_comparison.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Ensemble, energy ve knn kanıtlarını karşılaştırmak
- Açıklama: Pooled ve slice-aware OOD score yöntem karşılaştırmasını içerir.

## Test sample predictions
- Yol: `test/predictions.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Yanlış tahmin edilen örnekleri tek tek incelemek
- Açıklama: Her örnek için tahmin, etiket ve güven bilgisini CSV olarak sunar.

## Test hard example thumbnails
- Yol: `test/hard_examples_thumbnails`
- Öncelik: `medium`
- Format: `directory`
- Amaç: Zor örnekleri görsel olarak hızlıca taramak
- Açıklama: Zor örneklerin hızlı gözden geçirme için küçük önizlemelerini tutar.
