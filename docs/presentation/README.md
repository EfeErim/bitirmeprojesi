# Canva Presentation Assets

These SVG files were designed for the AADS v6 Canva presentation:

| Canva slide | Asset | Purpose |
| --- | --- | --- |
| 8 | `system_process.svg` | Overview of the full AADS v6 process |
| 13 | `leakage_safe_preparation.svg` | Leakage-aware grouped dataset preparation |
| 16 | `dinov3_sd_lora_architecture.svg` | DINOv3 and SD-LoRA architecture summary |
| 17 | `adapter_architecture_detail.svg` | Detailed AADS v6 adapter architecture with multi-scale fusion |
| 18 | `ood_vs_oe.svg` | Clear separation between OOD and OE roles |
| 20 | `readiness_result.svg` | Apricot fruit readiness result and deployment status |
| 21 | `router_guided_inference.svg` | Router-to-adapter flow with controlled abstention |

The source SVGs are Canva-ready. PNG exports and `presentation.pptx` can be regenerated on Windows with:

```powershell
.\scripts\python.cmd scripts\generate_presentation.py
```

The generator uses CairoSVG or Inkscape when available and falls back to Edge or Chrome.
