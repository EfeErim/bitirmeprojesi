# Auto-extracted from colab_notebooks/9_presentation_recording_demo.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

from scripts.notebook_helpers.presentation_demo_helpers import render_presentation_demo

presentation_summary = render_presentation_demo(
    ANALYSIS_IMAGE_PATH,  # noqa: F821 - provided by the notebook namespace
    router_result=router_result,  # noqa: F821 - provided by the notebook namespace
    auto_result=auto_result,  # noqa: F821 - provided by the notebook namespace
)

presentation_summary
