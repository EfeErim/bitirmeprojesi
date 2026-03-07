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
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=image_tensor.device)

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
