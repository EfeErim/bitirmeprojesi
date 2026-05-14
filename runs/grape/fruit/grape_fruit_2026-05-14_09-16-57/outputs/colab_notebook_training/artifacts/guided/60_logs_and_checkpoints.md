# Log ve Checkpoint

Bu bölüm runtime logları ve checkpoint kayıtlarını listeler.

## Telemetry ozeti
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/telemetry/summary.json`
- Öncelik: `high`
- Format: `json`
- Amaç: Notebook final ozetini okumak
- Açıklama: Notebook final ozet dosyasi.

## Telemetry event logu
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/telemetry/events.jsonl`
- Öncelik: `medium`
- Format: `jsonl`
- Amaç: Notebook akisini olay bazinda incelemek
- Açıklama: Notebook olayi bazli telemetry kaydi.

## Runtime logu
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/telemetry/runtime.log`
- Öncelik: `medium`
- Format: `log`
- Amaç: Calisma sirasindaki log ciktilarini okumak
- Açıklama: Notebook runtime boyunca yazilan metin logu.

## Best checkpoint manifesti
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/checkpoint_state/best_checkpoint.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Hangi checkpoint secildigini gormek
- Açıklama: En iyi checkpoint'in repo mirror manifesti.

## Latest checkpoint manifesti
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/checkpoint_state/latest_checkpoint.json`
- Öncelik: `medium`
- Format: `json`
- Amaç: Checkpoint akisini gormek
- Açıklama: Son checkpoint manifesti.

## Son durum ozeti
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/telemetry/latest_status.json`
- Öncelik: `low`
- Format: `json`
- Amaç: Kosunun son durumunu hizli kontrol etmek
- Açıklama: Notebook'un son durum snapshot'i.

## Checkpoint indexi
- Yol: `../../../runs/grape/fruit/grape_fruit_2026-05-14_09-16-57/checkpoint_state/checkpoint_index.json`
- Öncelik: `low`
- Format: `json`
- Amaç: Checkpoint kayitlarini toplu gormek
- Açıklama: Mirror edilen checkpoint manifest listesi.
