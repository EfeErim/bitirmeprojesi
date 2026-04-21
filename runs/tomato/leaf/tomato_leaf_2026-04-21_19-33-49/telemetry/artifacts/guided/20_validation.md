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

## Validation hard examples
- Yol: `validation/hard_examples.csv`
- Öncelik: `high`
- Format: `csv`
- Amaç: Feedback ile düzeltilecek zor örnekleri önceliklendirmek
- Açıklama: Yanlış sınıflanan veya kaçırılan OOD örneklerini öncelikli inceleme için listeler.

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

## Validation OOD tip kırılımı
- Yol: `validation/ood_type_breakdown.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Hangi OOD alt tipinde sorun olduğunu görmek
- Açıklama: OOD klasör alt tiplerine göre metrik kırılımını içerir.

## Validation OOD method comparison
- Yol: `validation/ood_method_comparison.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Ensemble, energy ve knn kanıtlarını karşılaştırmak
- Açıklama: Pooled ve slice-aware OOD score yöntem karşılaştırmasını içerir.

## Validation sample predictions
- Yol: `validation/predictions.csv`
- Öncelik: `medium`
- Format: `csv`
- Amaç: Yanlış tahmin edilen örnekleri tek tek incelemek
- Açıklama: Her örnek için tahmin, etiket ve güven bilgisini CSV olarak sunar.

## Validation hard example thumbnails
- Yol: `validation/hard_examples_thumbnails`
- Öncelik: `medium`
- Format: `directory`
- Amaç: Zor örnekleri görsel olarak hızlıca taramak
- Açıklama: Zor örneklerin hızlı gözden geçirme için küçük önizlemelerini tutar.
