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

## Optimization Strategy (V2)
To address performance gaps, the following protocols are being implemented:
1.  **Extended Convergence Horizon**: Increasing training to 100 epochs.
2.  **Targeted Geometric Augmentation**: Increased rotation, MixUp, and vertical flips.
3.  **Statistical Oversampling**: Balancing minority classes like Anthracnose Fruit Rot.
4.  **Hyperparameter Evolution**: Genetic Algorithm search for LoRA-specific parameters (optional).

## Installation
*(Instructions to be added)*

## Usage
*(Instructions to be added)*
