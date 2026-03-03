import pytest

from src.training.quantization import (
    HybridINT8Config,
    assert_no_prohibited_4bit_flags,
    find_prohibited_4bit_flags,
    load_hybrid_int8_backbone,
)


class DummyModel:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return {'model_name': model_name, 'kwargs': kwargs}


def test_find_prohibited_flags_detects_low_bit_settings():
    hits = find_prohibited_4bit_flags({'quantization': {'load_in_4bit': True}})
    assert hits


def test_find_prohibited_flags_detects_notebook_and_nested_surfaces():
    config = {
        'base': {'training': {'continual': {'quantization': {'mode': 'int8_hybrid'}}}},
        'colab': {'injected': {'bnb_4bit_quant_type': 'nf4'}},
        'notebook_runtime': [{'config_patch': {'load_in_4bit': False}}],
    }
    hits = find_prohibited_4bit_flags(config)
    assert any('colab.injected.bnb_4bit_quant_type' in path for path in hits)
    assert any('notebook_runtime' in path for path in hits)


def test_assert_no_prohibited_flags_rejects_low_bit_settings():
    with pytest.raises(ValueError):
        assert_no_prohibited_4bit_flags({'nf4_enabled': True})


def test_hybrid_config_validate_requires_int8_mode():
    cfg = HybridINT8Config(mode='fp16')
    with pytest.raises(ValueError):
        cfg.validate()


def test_int8_loader_fails_when_backend_missing_and_fallback_disabled(monkeypatch):
    from src.training import quantization as q

    monkeypatch.setattr(q, '_has_module', lambda _: False)
    cfg = HybridINT8Config(mode='int8_hybrid', strict_backend=True, allow_cpu_fallback=False)

    with pytest.raises(RuntimeError):
        load_hybrid_int8_backbone('facebook/dinov3-vitl16-pretrain-lvd1689m', auto_model_cls=DummyModel, cfg=cfg)


def test_int8_loader_allows_non_quantized_fallback_when_enabled(monkeypatch):
    from src.training import quantization as q

    monkeypatch.setattr(q, '_has_module', lambda _: False)
    cfg = HybridINT8Config(mode='int8_hybrid', strict_backend=False, allow_cpu_fallback=True)

    model = load_hybrid_int8_backbone('facebook/dinov3-vitl16-pretrain-lvd1689m', auto_model_cls=DummyModel, cfg=cfg)

    assert model['model_name'] == 'facebook/dinov3-vitl16-pretrain-lvd1689m'


def test_int8_loader_allows_non_quantized_fallback_with_strict_backend_if_explicitly_enabled(monkeypatch):
    from src.training import quantization as q

    monkeypatch.setattr(q, '_has_module', lambda _: False)
    cfg = HybridINT8Config(mode='int8_hybrid', strict_backend=True, allow_cpu_fallback=True)

    model = load_hybrid_int8_backbone('facebook/dinov3-vitl16-pretrain-lvd1689m', auto_model_cls=DummyModel, cfg=cfg)

    assert model['kwargs'] == {}

