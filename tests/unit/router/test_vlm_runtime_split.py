from types import SimpleNamespace

import torch
from PIL import Image

from src.router import clip_runtime, sam3_runtime
from src.router.vlm_pipeline import VLMPipeline


class FakeTokenizer:
    def __call__(self, prompts):
        return torch.arange(len(prompts), dtype=torch.float32).unsqueeze(1)


class FakeOpenClipModel:
    def __init__(self):
        self.encode_image_calls = 0
        self.logit_scale = torch.tensor(0.0)

    def encode_image(self, image_tensor):
        self.encode_image_calls += 1
        batch = int(image_tensor.shape[0])
        embeds = torch.zeros(batch, 2, dtype=torch.float32, device=image_tensor.device)
        embeds[:, 0] = 1.0
        return embeds

    def encode_text(self, text_tokens):
        prompt_count = int(text_tokens.shape[0])
        embeds = torch.zeros(prompt_count, 2, dtype=torch.float32, device=text_tokens.device)
        if prompt_count > 0:
            embeds[0] = torch.tensor([1.0, 0.0], dtype=torch.float32, device=text_tokens.device)
        if prompt_count > 1:
            embeds[1:] = torch.tensor([0.0, 1.0], dtype=torch.float32, device=text_tokens.device)
        return embeds


def _build_pipeline() -> VLMPipeline:
    return VLMPipeline(
        config={
            "vlm_enabled": True,
            "router": {
                "crop_mapping": {"tomato": {"parts": ["leaf"]}},
                "vlm": {
                    "enabled": True,
                    "crop_labels": ["tomato", "potato"],
                    "part_labels": ["leaf"],
                    "prompt_templates": {
                        "crop": ["{term}", "close {term}"],
                        "part": ["{term}"],
                    },
                },
            },
        },
        device="cpu",
    )


def _attach_open_clip_runtime(pipeline: VLMPipeline) -> FakeOpenClipModel:
    model = FakeOpenClipModel()
    pipeline.bioclip_backend = "open_clip"
    pipeline.bioclip = model
    pipeline.bioclip_processor = {
        "preprocess": lambda _image: torch.ones(3, 4, 4, dtype=torch.float32),
        "tokenizer": FakeTokenizer(),
    }
    return model


def test_clip_score_labels_ensemble_reuses_open_clip_image_embedding():
    pipeline = _build_pipeline()
    model = _attach_open_clip_runtime(pipeline)

    best_label, best_score, scores = clip_runtime.clip_score_labels_ensemble(
        pipeline,
        Image.new("RGB", (8, 8), color="green"),
        ["tomato", "potato"],
        label_type="crop",
        num_prompts=2,
    )

    assert best_label in {"tomato", "potato"}
    assert best_score >= 0.0
    assert set(scores.keys()) == {"tomato", "potato"}
    assert model.encode_image_calls == 1


def test_clip_score_labels_ensemble_batch_reuses_open_clip_image_embeddings():
    pipeline = _build_pipeline()
    model = _attach_open_clip_runtime(pipeline)

    results = clip_runtime.clip_score_labels_ensemble_batch(
        pipeline,
        [Image.new("RGB", (8, 8), color="green") for _ in range(3)],
        ["tomato", "potato"],
        label_type="crop",
        num_prompts=2,
        image_batch_size=8,
    )

    assert len(results) == 3
    assert all(result[0] in {"tomato", "potato"} for result in results)
    assert model.encode_image_calls == 1


def test_open_set_debug_payload_is_lazy_when_debug_disabled(monkeypatch):
    pipeline = _build_pipeline()
    _attach_open_clip_runtime(pipeline)

    monkeypatch.setattr(clip_runtime.logger, "isEnabledFor", lambda _level: False)
    monkeypatch.setattr(
        clip_runtime.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("debug payload should stay lazy")),
    )

    label, confidence = clip_runtime.clip_score_labels(
        pipeline,
        Image.new("RGB", (8, 8), color="green"),
        ["tomato"],
        label_type="crop",
    )

    assert label in {"tomato", "unknown"}
    assert confidence >= 0.0


def test_analyze_image_sam3_delegates_to_runtime_helpers(monkeypatch):
    pipeline = _build_pipeline()
    expected = {"detections": [{"crop": "tomato"}], "image_size": (3, 8, 8)}
    sentinel_context = SimpleNamespace()

    def fake_build_request_context(runtime, **kwargs):
        assert runtime is pipeline
        assert kwargs["image_size"] == (3, 8, 8)
        return sentinel_context

    monkeypatch.setattr(sam3_runtime, "build_request_context", fake_build_request_context)
    monkeypatch.setattr(sam3_runtime, "analyze_sam3_image", lambda runtime, context: expected)

    result = pipeline._analyze_image_sam3(Image.new("RGB", (8, 8), color="green"), (3, 8, 8))

    assert result == expected


def test_route_batch_preserves_order_and_uses_batched_runtime(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"

    analyses = [
        {"detections": [{"crop": "tomato", "part": "leaf", "bbox": [0, 0, 1, 1], "crop_confidence": 0.9}]},
        {"detections": [{"crop": "potato", "part": "leaf", "bbox": [1, 1, 2, 2], "crop_confidence": 0.8}]},
        {"detections": [{"crop": "pepper", "part": "leaf", "bbox": [2, 2, 3, 3], "crop_confidence": 0.7}]},
    ]

    monkeypatch.setattr(sam3_runtime, "analyze_sam3_batch", lambda runtime, batch: analyses)
    monkeypatch.setattr(
        pipeline,
        "analyze_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("per-image path should not run")),
    )

    crops_out, confs = pipeline.route_batch(torch.zeros(3, 3, 8, 8))

    assert [item["crop"] for item in crops_out] == ["tomato", "potato", "pepper"]
    assert confs == [0.9, 0.8, 0.7]


def test_analyze_sam3_batch_reduces_sam_call_count_under_batch_support(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"
    pipeline.vlm_config["batch_chunk_size"] = 3

    batch_calls = {"count": 0}
    fallback_calls = {"count": 0}

    def fake_run_sam3_batch(images, prompt, threshold=0.7):
        batch_calls["count"] += 1
        return [{"masks": [], "boxes": [], "scores": []} for _ in images]

    monkeypatch.setattr(pipeline, "_run_sam3_batch", fake_run_sam3_batch)
    monkeypatch.setattr(
        sam3_runtime,
        "analyze_sam3_image",
        lambda *_args, **_kwargs: fallback_calls.__setitem__("count", fallback_calls["count"] + 1),
    )

    results = sam3_runtime.analyze_sam3_batch(pipeline, torch.zeros(5, 3, 8, 8))

    assert len(results) == 5
    assert batch_calls["count"] == 2
    assert fallback_calls["count"] == 0


def test_analyze_sam3_batch_falls_back_per_chunk_when_batched_sam3_is_unsupported(monkeypatch):
    pipeline = _build_pipeline()
    pipeline.models_loaded = True
    pipeline.actual_pipeline = "sam3"
    pipeline.vlm_config["batch_chunk_size"] = 2

    monkeypatch.setattr(
        pipeline,
        "_run_sam3_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("batch unsupported")),
    )

    calls = {"count": 0}

    def fake_analyze_sam3_image(_runtime, context):
        index = calls["count"]
        calls["count"] += 1
        return {
            "detections": [
                {
                    "crop": f"crop_{index}",
                    "part": "leaf",
                    "bbox": [0, 0, 1, 1],
                    "crop_confidence": 0.5 + index,
                }
            ],
            "image_size": context.image_size,
        }

    monkeypatch.setattr(sam3_runtime, "analyze_sam3_image", fake_analyze_sam3_image)

    results = sam3_runtime.analyze_sam3_batch(pipeline, torch.zeros(3, 3, 8, 8))

    assert len(results) == 3
    assert [result["detections"][0]["crop"] for result in results] == ["crop_0", "crop_1", "crop_2"]
    assert calls["count"] == 3


def test_set_runtime_profile_refreshes_profile_derived_controls():
    pipeline = VLMPipeline(
        config={
            "router": {
                "vlm": {
                    "enabled": True,
                    "open_set_enabled": False,
                    "profiles": {
                        "disabled": {"enabled": False},
                        "open_set": {"open_set_enabled": True},
                    },
                },
            },
        },
        device="cpu",
    )

    assert pipeline.enabled is True
    assert pipeline.open_set_enabled is False

    assert pipeline.set_runtime_profile("disabled") is True
    assert pipeline.enabled is False

    assert pipeline.set_runtime_profile("open_set") is True
    assert pipeline.open_set_enabled is True
