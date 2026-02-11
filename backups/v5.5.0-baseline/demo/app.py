#!/usr/bin/env python3
"""
Gradio Demo for AADS-ULoRA v5.5
Interactive interface for crop disease diagnosis with OOD detection.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import logging
from typing import Dict, Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.router.simple_crop_router import SimpleCropRouter
from src.ood.dynamic_thresholds import DynamicOODThreshold
from src.utils.data_loader import preprocess_image

logger = logging.getLogger(__name__)

class AADSDemo:
    """Main demo class integrating all components"""
    
    def __init__(
        self,
        adapter_dir: str = './adapters',
        router_path: str = './router/crop_router_best.pth',
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load components
        logger.info("Loading crop router...")
        self.router = self._load_router(router_path)
        
        logger.info("Loading crop adapters...")
        self.adapters = self._load_adapters(adapter_dir)
        
        logger.info("Demo initialized successfully!")
    
    def _load_router(self, router_path: str) -> SimpleCropRouter:
        """Load trained crop router"""
        # Implementation depends on router training
        # For now, return a placeholder
        return SimpleCropRouter(crops=['tomato', 'pepper', 'corn'])
    
    def _load_adapters(self, adapter_dir: str) -> Dict[str, IndependentCropAdapter]:
        """Load all trained crop adapters"""
        adapters = {}
        adapter_dir = Path(adapter_dir)
        
        for crop in ['tomato', 'pepper', 'corn']:
            crop_adapter_path = adapter_dir / crop
            if crop_adapter_path.exists():
                logger.info(f"Loading {crop} adapter...")
                adapter = IndependentCropAdapter(crop_name=crop)
                adapter.load_adapter(str(crop_adapter_path))
                adapters[crop] = adapter
        
        return adapters
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess uploaded image for model input"""
        return preprocess_image(image, target_size=224)
    
    def predict(
        self, 
        image: Image.Image, 
        crop_hint: Optional[str] = None
    ) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Input PIL image
            crop_hint: Optional crop type hint
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_tensor = self.preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Step 1: Crop routing
        if crop_hint and crop_hint in self.adapters:
            predicted_crop = crop_hint
            crop_confidence = 1.0
        else:
            predicted_crop, crop_confidence = self.router.route(img_tensor)
        
        # Step 2: Get appropriate adapter
        if predicted_crop not in self.adapters:
            return {
                'status': 'error',
                'message': f'No adapter available for crop: {predicted_crop}'
            }
        
        adapter = self.adapters[predicted_crop]
        
        # Step 3: Disease prediction with OOD detection
        result = adapter.predict_with_ood(img_tensor)
        
        # Add crop info
        result['crop'] = predicted_crop
        result['crop_confidence'] = crop_confidence
        
        return result
    
    def create_ui(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="AADS-ULoRA v5.5 Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# AADS-ULoRA v5.5: Agricultural Disease Diagnosis")
            gr.Markdown("Upload an image of a plant leaf/fruit to detect diseases with dynamic OOD detection.")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    crop_hint = gr.Dropdown(
                        choices=["auto", "tomato", "pepper", "corn"],
                        value="auto",
                        label="Crop Type (optional hint)"
                    )
                    submit_btn = gr.Button("Diagnose", variant="primary")
                
                with gr.Column():
                    output_text = gr.JSON(label="Diagnosis Results")
            
            # Examples
            gr.Markdown("## Example Images")
            examples = gr.Examples(
                examples=[
                    ["examples/tomato_healthy.jpg", "tomato"],
                    ["examples/tomato_early_blight.jpg", "tomato"],
                    ["examples/pepper_healthy.jpg", "pepper"]
                ],
                inputs=[image_input, crop_hint],
                label="Try these examples"
            )
            
            # Event handler
            submit_btn.click(
                fn=self.predict,
                inputs=[image_input, crop_hint],
                outputs=output_text
            )
            
            # Info section
            with gr.Accordion("System Information", open=False):
                gr.Markdown(f"""
                **System Components:**
                - Crop Router: Routes images to correct crop adapter
                - Independent Crop Adapters: One per crop type
                - Dynamic OOD Detection: Mahalanobis distance with per-class thresholds
                
                **Supported Crops:** {', '.join(self.adapters.keys())}
                """)
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Gradio demo"""
        demo = self.create_ui()
        demo.launch(**kwargs)


def main():
    """Run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_dir', type=str, default='./adapters')
    parser.add_argument('--router_path', type=str, default='./router/crop_router_best.pth')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and launch demo
    demo = AADSDemo(
        adapter_dir=args.adapter_dir,
        router_path=args.router_path
    )
    
    logger.info(f"Starting Gradio demo on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name='0.0.0.0'
    )


if __name__ == "__main__":
    main()