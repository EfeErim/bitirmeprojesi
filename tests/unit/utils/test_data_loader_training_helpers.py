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
