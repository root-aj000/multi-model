# Multi-Model Prediction System — Project Guide

> Fine-Grained Multi-Modal Fusion Network (FG_MFN) for multi-attribute ad creative classification

This guide covers the complete documentation for the Multi-Model Prediction System — a deep learning application that combines visual and text features to classify ad creatives across 9 attributes simultaneously.

## Guide Structure

| Document | Description |
|----------|-------------|
| [01 — Quick Start](01_quick_start.md) | Installation, configuration, and running the system in 5 minutes |
| [02 — Architecture](02_architecture.md) | System architecture, module relationships, and design patterns |
| [03 — Model Reference](03_model_reference.md) | FG_MFN model architecture, sub-modules, and fusion strategies |
| [04 — API Reference](04_api_reference.md) | FastAPI endpoints, request/response schemas, and usage examples |
| [05 — Preprocessing](05_preprocessing.md) | Text cleaning, tokenization, image transforms, and dataset loading |
| [06 — OCR Engines](06_ocr_engines.md) | EasyOCR and PaddleOCR integration, factory pattern, and configuration |
| [07 — Training Guide](07_training_guide.md) | Training pipeline, Mixup augmentation, warmup scheduling, and checkpointing |
| [08 — Evaluation Guide](08_evaluation_guide.md) | Model evaluation, metrics computation, and result persistence |
| [09 — CLI Scripts](09_cli_scripts.md) | Command-line tools for training, evaluation, serving, and analysis |
| [10 — Configuration](10_configuration.md) | Complete configuration schema with all supported keys and defaults |
| [11 — Utilities](11_utilities.md) | Cache management, logging, lifecycle, and config loading utilities |
| [12 — Testing](12_testing.md) | Test suite structure, running tests, and writing new tests |
| [13 — Dataflow Diagrams](13_dataflow_diagrams.md) | Visual dataflow documentation for every function in the codebase |

## Quick Links

- **Start here:** [01 — Quick Start](01_quick_start.md)
- **Understand the model:** [03 — Model Reference](03_model_reference.md)
- **Deploy the API:** [04 — API Reference](04_api_reference.md)
- **Train a model:** [07 — Training Guide](07_training_guide.md)
- **Visualize dataflow:** [13 — Dataflow Diagrams](13_dataflow_diagrams.md)
