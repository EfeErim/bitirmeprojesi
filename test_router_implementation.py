from src.router.simple_crop_router import SimpleCropRouter
import torch

# Test router initialization
crops = ['tomato', 'pepper', 'corn']
router = SimpleCropRouter(
    crops=crops,
    model_name='facebook/dinov3-base',
    device='cpu',
    confidence_threshold=0.92,
    top_k_alternatives=3
)

print("Router initialized successfully!")
print(f"Confidence threshold: {router.confidence_threshold}")
print(f"Top-K alternatives: {router.top_k_alternatives}")
print(f"Confidence stats: {router.confidence_stats}")

# Test confidence statistics methods
print("\nTesting confidence statistics:")
print(f"Initial stats: {router.get_confidence_stats()}")
print(f"Full stats: {router.get_full_stats()}")

# Test cache stats
print("\nTesting cache statistics:")
print(f"Cache stats: {router.get_cache_stats()}")

print("\nAll tests passed!")