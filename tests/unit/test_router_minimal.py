from src.router.vlm_pipeline import VLMPipeline

# Test VLM pipeline initialization
config = {
    'vlm_enabled': True,
    'vlm_confidence_threshold': 0.8,
    'vlm_max_detections': 10
}
pipeline = VLMPipeline(
    config=config,
    device='cpu'
)

print("VLM Pipeline initialized successfully!")
pipeline.load_models()

# Test processing an image
dummy_input = torch.randn(1, 3, 224, 224)
result = pipeline.process_image(dummy_input)

print(f"\nProcessing result status: {result['status']}")
print(f"Scenario: {result['scenario']}")

print("\nAll tests passed!")