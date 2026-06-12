# ROI, BBox, Router ve Adapter Calisma Notu

Son guncelleme: 2026-06-12

Bu belge, part-aware ROI/bbox ablation hattinda ne denendigini, neyin calistigini, neyin calismadigini ve bundan sonra hangi sirayla ilerlenmesi gerektigini tek yerde tutar. Konu dagildiginda once bu belgeye bak.

## Kisa Durum

- Baslangic fikri: router/SAM benzeri bbox kullanarak adapterin arka plana bakmasini azaltmak ve adaptere hedef parca ROI crop gostermek.
- Kapsam daha sonra maske/segmentasyon degil, bbox tabanli ROI crop olarak netlesti.
- Ilk deneyler whole-image trained adapterlara inference zamaninda ROI crop vermenin iyi olup olmadigini olctu.
- Sonuc: bbox bulmak tek basina yetmiyor. Whole-image trained adapter, crop input gorunce distribution shift yasiyor ve performans dusuyor.
- Su an en guvenli calisma modu: adapter final karari full image uzerinden verir; router/Grounding DINO ROI sadece evidence/review sinyali olarak kullanilir.
- Yeniden egitim gerekiyorsa inference view policy ile training view policy ayni olmalidir. Sadece inference'da crop vermek bilimsel olarak temiz degil.
- Daha ileri adapter retrain icin dataset tarafinda gercek bbox annotation gerekir. Pseudo bbox ile training yapilabilir ama label noise riski yuksektir.

## Yapilanlar

### Historical Notebook 10-14 ROI Ablation Yuzeyi

- `10_ablation_full_image_baseline.ipynb`: full image baseline.
- `11_ablation_primary_roi_inference.ipynb`: router primary bbox ile saf ROI inference.
- `12_ablation_hybrid_roi_fallback.ipynb`: bbox uygunsa ROI, yoksa full fallback.
- `13_ablation_roi_trained_adapter.ipynb`: ROI-trained adapter ikinci faz yuzeyi.
- `14_ablation_mixed_full_roi_training.ipynb`: full + ROI mixed training deneyi.
- Ortak mantik `scripts/colab_roi_ablation.py` icinde tutuldu.
- Raporlar `docs/ablation_results/<condition>/` altina yaziliyor ve Colab'den repoya pushlanabiliyor.
- Bu wrapper notebooklar sonradan silindi. Aktif ROI/bbox notebook yuzeyi yalnizca `16_ablation_dual_view_inference.ipynb`; eski rapor klasorleri historical evidence olarak tutuluyor.

### Grounding DINO Target-Aware ROI

- Notebook 16'ya `router_then_grounding_dino` target ROI backend eklendi.
- Amac: router `tomato__fruit` hedefini bulamazsa text-guided detector ile `tomato fruit` bbox aramak.
- Grounding DINO promptlari Hugging Face formatina cekildi: lowercase ve nokta ile biten query.
- DINO hata/status alanlari rapora eklendi:
  - `grounding_dino_status`
  - `grounding_dino_error`
  - `grounding_dino_candidate_count`
  - `target_detection_source`

### Notebook 16 Son Bulgulari

Son calisan Notebook 16 raporunda:

- `target_detection_found`: `119/119`
- `target_detection_source`:
  - `grounding_dino`: `99`
  - `router_detection`: `20`
- `grounding_dino_status`: `ok` olan `99` satir
- `grounding_dino_error`: bos
- `grounding_dino_candidate_count` toplam: `3140`
- `roi_quality_status`:
  - `roi_ok`: `87`
  - `roi_too_large`: `31`
  - `roi_too_small`: `1`

Performans:

- Notebook 16 yeni accuracy: `0.8655`
- macro-F1: `0.8256`
- Ayni satirlarda full-image prediction accuracy: `0.8992`
- ROI available iken ROI prediction accuracy: `0.8276`
- ROI secilen 15 ornekte accuracy: `0.6000`

Yorum:

- Problem artik "bbox bulunamiyor" degil.
- Grounding DINO bbox buluyor.
- Asil problem: whole-image trained adaptere crop input vermek performansi bozuyor.
- Bbox crop bazen hastalik sinyalini, bitki baglamini veya egitim dagilimini eksiltiyor.

### Notebook 16 Karari

Notebook 16 artik score-fusion/ROI override yuzeyi degil.

Yeni politika:

- `decision_policy = full_image_primary_with_roi_evidence`
- Final tahmin her zaman full-image adapter prediction.
- ROI prediction varsa sadece evidence olarak raporlanir.
- ROI ile full image celisirse sonuc degismez; `requires_review=True` yazilir.
- Review policy sonradan sikilastirildi: `roi_too_large`, `roi_too_small`, `semantic_mismatch`, `roi_conflict` ve `roi_confidence_leads` tek basina review nedeni degildir. Bu sinyaller yalnizca `low_full_confidence` ile birlesince review reason olarak yazilir. `target_detection_missing` ve `grounding_dino_error` dogrudan review nedenidir.
- Yeni alanlar:
  - `final_view`
  - `roi_evidence_status`
  - `requires_review`
  - `review_reasons`
  - `full_confidence_review_threshold`
- Summary alanlari:
  - `requires_review_rate`
  - `roi_conflict_rate`
  - `review_capture_rate_on_errors`
  - `review_false_positive_rate_on_correct`

## Net Kararlar

1. ROI crop, mevcut whole-image adapter icin final karar override'i olmayacak.
2. Router/Grounding DINO bbox, simdilik adapter inputunu degistirmek icin degil evidence/review icin kullanilacak.
3. ROI-only adapter training tek basina ana aday degil; crop distribution shift ve label noise riski yuksek.
4. Yeniden egitim yapilacaksa train/val/test/inference view policy ayni olacak.
5. Gercek bbox annotation olmadan "adapteri bboxlara gore egitmek" zayif denetimli/pseudo-label training olur; bu acikca ayrilmali.
6. Hastalik sinifi image-level label ise bbox crop label'i otomatik dogru kabul edilmeyecek. Crop icinde semptom yoksa label transfer noise uretir.

## Data Tarafi: Neden El Atmak Gerekiyor

Adapteri bboxlara gore egitmek icin data tarafinda annotation gerekir.

Minimum annotation semasi:

- `image_path`
- `class_label`
- `crop`
- `part`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `bbox_type`: `plant_part`, `symptom_region`, `whole_plant`
- `annotator`
- `quality`: `ok`, `uncertain`, `multi_object`, `occluded`, `label_transfer_unsafe`

Kritik ayrim:

- `plant_part` bbox: domates meyvesi/yaprak/sap gibi parcanin yeri.
- `symptom_region` bbox: hastalik belirtisinin gorundugu bolge.
- `whole_plant` bbox: tum bitki.

Hastalik adapteri icin en mantikli ilk hedef `symptom_region + context padding` veya semptom gorunmuyorsa `plant_part + label_transfer_unsafe` flag'idir. Sadece `tomato fruit` bbox'i her zaman anthracnose gibi label'i tasimayabilir.

## Sonraki Is Sirasi

### 1. Notebook 16'yi Yeni Evidence Modunda Tekrar Calistir

Amac accuracy artirmak degil, review sinyalinin degerini olcmek:

- Full-image accuracy korunuyor mu?
- Yanlis tahminler `requires_review=True` ile yakalaniyor mu?
- Dogru tahminlerde gereksiz review orani cok yuksek mi?
- `roi_evidence_status=conflicts_with_full` gercek hatalarla korele mi?

Bakilacak alanlar:

- `accuracy`
- `macro_f1`
- `requires_review_rate`
- `roi_conflict_rate`
- `review_capture_rate_on_errors`
- `review_false_positive_rate_on_correct`
- `target_detection_source`
- `grounding_dino_status`
- `roi_quality_status`

### 2. BBox Annotation Pilot Set Hazirla

Amac tum datayi hemen etiketlemek degil, once policy'yi dogrulamak.

Pilot onerisi:

- Train/val/test icinden dengeli kucuk set.
- Sinif basina yaklasik 20-30 image.
- En az iki bbox tipi ayrimi:
  - `plant_part`
  - `symptom_region`
- Her bbox icin `quality` ve `label_transfer_safe/unsafe` bilgisi.

### 3. Annotation Manifest Uzerinden Domain-Shift-Safe Training

Yeni training view policy:

- Full image
- Annotated bbox crop
- Padded bbox crop, ornek: `pad_ratio=0.08`
- Context-preserving padded bbox crop, ornek: `pad_ratio=0.20`

Inference view policy de ayni view ailesini kullanmali:

- Full image prediction
- Annotated-policy benzeri router/Grounding DINO bbox prediction
- Padded/context ROI prediction
- Final karar:
  - once full-primary + evidence gate
  - sonra gerekirse calibrated/learned aggregator

### 4. Adapter Retrain Deneyi

Gelecekte eklenecek yeni shared helper/notebook, gercek annotation manifesti ile su metrikleri uretmeli:

- full-only accuracy
- ROI-only accuracy
- padded/context ROI accuracy
- full-primary evidence-gated review metrics
- full+ROI calibrated aggregation accuracy
- OOD/readiness yeniden kalibrasyonu
- `production_readiness.json` yeniden uretimi

### 5. Router Tarafi

Router icin yapilacaklar adapter retrain'den ayridir:

- Router hedef bitki/parcayi dogru ariyor mu?
- Router primary detection yanlis bitki ise target-aware detection kullaniliyor mu?
- Grounding DINO fallback sadece router hedefi bulamadiginda calisiyor mu?
- Bbox kalite filtresi `roi_too_large` durumlarini iyi yakaliyor mu?
- Inference bbox tipi training bbox tipiyle uyumlu mu?

## Yapilmayacaklar

- Whole-image trained adaptere sadece inference'da hard ROI crop verip sonucu final kabul etmek.
- Grounding DINO pseudo bbox'larini gercek annotation gibi sessizce training label yapmak.
- ROI-only training sonucunu full-image baseline ile ayni domain kosuluymus gibi yorumlamak.
- Tek global accuracy ile karar vermek; part/crop, bbox quality, review capture ve OOD/readiness birlikte bakilmali.

## Kisa Sonuc

Bu hattin dogru yonu:

1. Kisa vadede full-image adapter final karar + ROI evidence/review.
2. Orta vadede bbox annotation pilotu.
3. Sonra domain-shift-safe multi-view adapter retrain.
4. En son full+ROI aggregation veya learned gate.

Ana prensip:

Inference'da adaptere hangi view gosterilecekse, training'de de ayni view ailesi gosterilecek. Aksi halde tekrar ayni domain shift problemi yasanir.

## Sistemi Calistirmaya Goturen Roadmap

Bu roadmap rapor/sunum odakli degil, sistemin gercek kullanimda daha guvenilir calismasi icindir.

### 1. Current Best Pipeline'i Sabitle

Su an production'a en yakin aday:

```text
image
-> router crop/part
-> adapter full-image prediction
-> router/Grounding DINO bbox evidence
-> review gate
-> OOD/readiness gate
-> final response
```

Kararlar:

- Adapter final tahmini full image uzerinden verir.
- ROI/bbox adapter sonucunu override etmez.
- ROI sadece evidence/review sinyali olur.
- Notebook 16 bu davranisin test yuzeyidir ve tek `tomato__fruit` hedefe bagli degildir; matching prepared dataset + adapter export ciftlerini otomatik kesfedip coklu adapter hedefi olarak kosabilir.

Basari kriteri:

- Full-image accuracy korunur.
- Review rate makul kalir.
- Yanlislarin anlamli kismi review'a duser.
- Grounding DINO error uretmez.

### 2. Router Dogrulugunu Artir

Router yanlis bitki/parca ararsa adapter veya bbox kalitesi tek basina sistemi kurtarmaz.

Yapilacaklar:

- Router eval set olustur:
  - `image_path`
  - `expected_crop`
  - `expected_part`
  - `target_object_present`
- Router metrikleri:
  - crop accuracy
  - part accuracy
  - target bbox found rate
  - wrong crop rate
  - unknown/abstain rate
- Router yanlis crop verirse adapter cagrilmaz.
- Crop dogru ama part belirsizse explicit fallback/abstain policy uygulanir.
- Grounding DINO sadece target-aware fallback olarak kullanilir.

Basari kriteri:

- Wrong adapter load minimuma iner.
- `part=unknown` sessizce adapter inference'a gitmez.
- Router/Grounding DINO target-compatible bbox evidence uretir.

### 3. Evidence Gate'i Kalibre Et

Mevcut gate elle yazilmis konservatif policy'dir:

- `low_full_confidence`
- `target_detection_missing`
- `grounding_dino_error`
- `low_full_confidence + roi_conflict`
- `low_full_confidence + roi_too_large`

Yapilacaklar:

- Validation set uzerinde threshold search:
  - full confidence threshold
  - OOD score threshold
  - bbox area thresholds
  - ROI/full disagreement threshold
- Optimize edilecek hedef:
  - yanlis tahmin yakalama orani yuksek
  - gereksiz review orani dusuk
- Threshold'lar config-driven olacak; hardcoded kalmayacak.

Basari kriteri:

- Review rate kontrollu.
- Wrong capture rate yuksek.
- Threshold'lar val/test ayrimiyla olculmus.

### 4. Data/BBox Annotation Pilot

Adapteri bbox'a gore egitmek isteniyorsa gercek annotation gerekir. Pseudo bbox ana training label'i gibi kullanilmayacak.

Yapilacaklar:

- Kucuk ama temiz bbox pilot set:
  - sinif basina yaklasik 20-30 image
  - split bilgisi korunmus train/val/test secimi
- Annotation tipleri:
  - `plant_part`
  - `symptom_region`
  - `whole_plant`
- Annotation kalite alanlari:
  - `ok`
  - `uncertain`
  - `occluded`
  - `multi_object`
  - `label_transfer_unsafe`

Basari kriteri:

- Bbox crop gercekten image label'ini tasiyor mu olculur.
- Semptom gorunmeyen crop training'e sokulmaz veya `label_transfer_unsafe` olur.
- Hangi bbox tipinin ise yaradigi anlasilir.

### 5. Domain-Shift-Safe Adapter Retrain

Tekrar egitim yapilacaksa kural:

```text
training view policy == inference view policy
```

Training view set:

- full image
- padded plant-part bbox
- padded symptom bbox
- context ROI

Inference view set ayni aileyi kullanir:

- full image prediction
- bbox/context ROI prediction
- evidence aggregation
- review/OOD gate

Ilk retrain stratejisi:

- ROI-only degil.
- Multi-view trained adapter.
- Final karar baslangicta yine full-primary.
- ROI prediction once calibrated evidence olarak denenir.

Basari kriteri:

- Full-only baseline dusmez.
- Multi-view model full baseline'i gecer veya yanlislari daha iyi flagler.
- OOD calibration yeniden yapilir.
- `production_readiness.json` yeniden uretilir.

### 6. OOD ve Readiness'i Yeniden Kalibre Et

Pipeline veya adapter degistikce OOD davranisi da degisebilir.

Yapilacaklar:

- Her ciddi adapter/pipeline degisiminden sonra:
  - OOD calibration
  - OE check
  - readiness report
  - `production_readiness.json`
- OOD sadece adapter confidence ile degil, router/evidence sinyalleriyle birlikte yorumlanir.

Basari kriteri:

- Known-class accuracy iyi.
- Unknown/OOD rejection mantikli.
- Readiness dosyasi yeni pipeline'a ait.

### 7. Production Runtime'a Tasi

Notebook sonucu iyi cikinca logic notebookta kalmayacak.

Yapilacaklar:

- Notebook 16 evidence gate logic'i canonical runtime'a tasinir:
  - uygun `src/pipeline` veya workflow katmani
- Notebook 16 sadece wrapper olur.
- Test senaryolari:
  - router wrong crop
  - bbox missing
  - Grounding DINO error
  - low confidence
  - ROI conflict
  - OOD
  - adapter missing

Basari kriteri:

- Colab notebook, CLI ve runtime ayni karari uretir.
- Production inference davranisi testle korunur.

### Net Siralama

1. Router eval set + router hata analizi.
2. Notebook 16 multi-adapter sonuclarindan adapter-bazli evidence gate threshold analizi.
3. Bbox annotation pilot manifest formati.
4. Kucuk annotated bbox pilotu.
5. Multi-view adapter retrain.
6. OOD/readiness recalibration.
7. Runtime entegrasyonu.

### En Yakin Aksiyon

Bir sonraki teknik is:

```text
Router + evidence gate evaluation analyzer
```

Yeni notebook acmadan once `scripts/colab_roi_ablation.py` icine aggregate analyzer eklemek daha temizdir. Notebook 16 sonucu uzerinden su ayrim yapilmalidir:

- router hatasi mi?
- bbox hatasi mi?
- adapter hatasi mi?
- confidence/OOD hatasi mi?

Bu ayrim yapilmadan retrain'e girilirse tekrar kor ilerlenir.
