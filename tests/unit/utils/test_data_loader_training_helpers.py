from pathlib import Path

from PIL import Image
import torch

from src.utils import data_loader
from src.utils.data_loader import create_training_loaders, dict_collate_fn


class DummyCropDataset:
    def __init__(self, *args, **kwargs):
        self.samples = [
            (torch.zeros(3, 8, 8), 0),
            (torch.ones(3, 8, 8), 1),
            (torch.full((3, 8, 8), 2.0), 0),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_dict_collate_fn_transforms_tuple_batch():
    batch = [
        (torch.zeros(3, 8, 8), 0),
        (torch.ones(3, 8, 8), 1),
    ]

    collated = dict_collate_fn(batch)

    assert set(collated.keys()) == {'images', 'labels'}
    assert collated['images'].shape == (2, 3, 8, 8)
    assert torch.equal(collated['labels'], torch.tensor([0, 1], dtype=torch.long))


def test_create_training_loaders_uses_dict_collation(monkeypatch):
    monkeypatch.setattr(data_loader, 'CropDataset', DummyCropDataset)

    loaders = create_training_loaders(
        data_dir='unused',
        crop='tomato',
        batch_size=2,
        num_workers=0,
        use_cache=False,
    )

    assert {'train', 'val', 'test'} <= set(loaders.keys())

    batch = next(iter(loaders['train']))
    assert set(batch.keys()) == {'images', 'labels'}
    assert batch['images'].ndim == 4
    assert batch['labels'].dtype == torch.long


def _write_image(path: Path, *, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_create_training_loaders_supports_weighted_sampler(tmp_path: Path):
    for idx in range(3):
        _write_image(tmp_path / "tomato" / "continual" / "healthy" / f"healthy_{idx}.jpg")
    _write_image(tmp_path / "tomato" / "continual" / "disease_a" / "disease.jpg", color=(0, 255, 0))
    _write_image(tmp_path / "tomato" / "val" / "healthy" / "val.jpg")
    _write_image(tmp_path / "tomato" / "test" / "healthy" / "test.jpg")

    loaders = create_training_loaders(
        data_dir=str(tmp_path),
        crop="tomato",
        batch_size=2,
        num_workers=0,
        sampler="weighted",
        seed=7,
    )

    assert loaders["train"].sampler is not None
    assert loaders["train"].sampler.__class__.__name__ == "WeightedRandomSampler"


def test_crop_dataset_strict_error_policy_rejects_invalid_images(tmp_path: Path):
    bad_image = tmp_path / "tomato" / "continual" / "healthy" / "bad.jpg"
    bad_image.parent.mkdir(parents=True, exist_ok=True)
    bad_image.write_bytes(b"not-an-image")

    try:
        data_loader.CropDataset(
            data_dir=str(tmp_path),
            crop="tomato",
            split="train",
            use_cache=False,
            error_policy="strict",
            validate_images_on_init=True,
        )
        assert False, "strict mode should reject invalid images"
    except ValueError as exc:
        assert "Failed to validate dataset image" in str(exc)


def test_crop_dataset_tolerant_error_policy_skips_invalid_images(tmp_path: Path):
    _write_image(tmp_path / "tomato" / "continual" / "healthy" / "good.jpg")
    bad_image = tmp_path / "tomato" / "continual" / "healthy" / "bad.jpg"
    bad_image.write_bytes(b"not-an-image")

    dataset = data_loader.CropDataset(
        data_dir=str(tmp_path),
        crop="tomato",
        split="train",
        use_cache=False,
        error_policy="tolerant",
        validate_images_on_init=True,
    )

    assert len(dataset) == 1
    stats = dataset.get_cache_stats()
    assert stats["load_error_count"] == 1
    assert any("bad.jpg" in item for item in stats["skipped_files"])
