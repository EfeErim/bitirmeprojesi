import numpy as np
from PIL import Image

from src.data.transforms import build_image_transform, preprocess_image


def test_build_image_transform_training_policy_contains_expected_transforms():
    transform = build_image_transform(target_size=224, training=True)
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "ColorJitter",
        "RandomGrayscale",
        "RandomApply",
        "RandomPerspective",
        "ToTensor",
        "Normalize",
        "RandomErasing",
    ]


def test_build_image_transform_inference_policy_is_deterministic():
    transform = build_image_transform(target_size=224, training=False)
    names = [step.__class__.__name__ for step in transform.transforms]

    assert names == ["Resize", "ToTensor", "Normalize"]


def test_preprocess_image_returns_expected_tensor_shape():
    image = Image.new("RGB", (320, 240), color=(120, 200, 80))
    tensor = preprocess_image(image, target_size=224)

    assert tensor.shape == (3, 224, 224)


def test_preprocess_image_supports_numpy_rgb_input():
    array = np.zeros((64, 96, 3), dtype=np.uint8)
    array[..., 0] = 255

    tensor = preprocess_image(array, target_size=224)

    assert tensor.shape == (3, 224, 224)
