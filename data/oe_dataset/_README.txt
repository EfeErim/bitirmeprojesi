AADS OE candidate pools generated from the user-provided outlier ZIP files.

Use these paths with Notebook 2 as OE_ROOT when OE_ENABLED=True, or copy one selected tree into a prepared runtime dataset as oe/.

Important:
- OE images are training-only auxiliary outliers.
- Do not reuse the same files as final ood/ readiness evidence in the same run.
- Leaf-adapter candidate pools that contain leaf diseases are only safe when those disease labels are not supported classes for that adapter.
- Review strawberry calciumdeficiency visually before using it for a fruit adapter if calcium deficiency is intended to become a supported fruit class.
