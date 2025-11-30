# Phase T1: Strawberry Domain Adaptation

## Project Overview
This project focuses on validating the "Universal Feature Extractor" hypothesis by training a lightweight Low-Rank Adapter (LoRA) for the Strawberry domain. The primary goal is to transfer geometric features (edges, textures, shapes) from a frozen T0 (Tomato) backbone to detect 7 new Strawberry disease classes without retraining the backbone.

## Dataset
- **Source**: Strawberry Disease Detection Dataset (Roboflow Universe)
- **Classes**: 7 distinct classes covering leaf and fruit pathologies.

## Current Status (Phase T1)
**Date**: November 23, 2025
**Module**: T1 Strawberry Adapter (`T1_Strawberry_Adapter.pt`)
**Base Architecture**: YOLOv8n (Frozen T0 Backbone)

### Quantitative Results
- **mAP@50**: 89.4%
- **Performance**: Achieved near-baseline performance (Tomato baseline: 93.5%) with a frozen backbone.
- **High Accuracy**: Leaf Spot (98%), Powdery Mildew Fruit (97%).
- **Areas for Improvement**: Anthracnose Fruit Rot (55%), Angular Leafspot (75%).

## Project Timeline (7-Week Plan)
| Week | Stage | Focus |
| :--- | :--- | :--- |
| **W1** | IM Development & PoC | Implement Iterative Merging Core, Train Strawberry Adapter (T1). |
| **W2** | Sequential Expansion (T1/T2) | Validate T1, Train Apricot Adapter (T2), Merge T1+T2. |
| **W3** | Sequential Expansion (T3) | Train Hazelnut Adapter (T3), Merge T3. |
| **W4** | N=5 Stress Test (T4) | Train Grape Adapter (T4), Final Merge (Model v5). |
| **W5** | Final Validation | Measure performance on all 5 domains. Constraint profiling (Size ≤6MB). |
| **W6** | Reporting & Analysis | Draft quantitative results and conclusion. |
| **W7** | Final Review | Final polish and thesis submission. |

## Optimization Strategy (V2)
To address performance gaps, the following protocols are being implemented:
1.  **Extended Convergence Horizon**: Increasing training to 100 epochs.
2.  **Targeted Geometric Augmentation**: Increased rotation, MixUp, and vertical flips.
3.  **Statistical Oversampling**: Balancing minority classes like Anthracnose Fruit Rot.
4.  **Hyperparameter Evolution**: Genetic Algorithm search for LoRA-specific parameters (optional).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/EfeErim/bitirmeprojesi.git
   cd bitirmeprojesi
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Google Colab
To set up the environment in Colab (mount Drive and install requirements), run the following in a cell:
```python
from utils.colab_setup import setup_project
setup_project()
```

