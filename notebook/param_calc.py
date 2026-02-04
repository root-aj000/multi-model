# notebook/param_calc.py - FIXED VERSION

import torch
import json
from models.fg_mfn import FG_MFN

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def analyze_model(cfg, num_samples=10000):
    """Analyze model size and suitability for dataset"""
    model = FG_MFN(cfg)
    total, trainable, frozen = count_parameters(model)
    
    print("=" * 60)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Image Backbone: {cfg['IMAGE_BACKBONE']}")
    print(f"  Text Encoder: {cfg['TEXT_ENCODER']}")
    print(f"  Hidden Dim: {cfg['HIDDEN_DIM']}")
    print(f"  Freeze Backbone: {cfg.get('FREEZE_BACKBONE', False)}")
    
    print(f"\nParameter Counts:")
    print(f"  Total Parameters:     {total:>15,}")
    print(f"  Trainable Parameters: {trainable:>15,}")
    print(f"  Frozen Parameters:    {frozen:>15,}")
    
    # Suitability analysis - CORRECTED for transfer learning
    is_frozen = cfg.get('FREEZE_BACKBONE', False)
    
    print(f"\nDataset Suitability (for {num_samples:,} samples):")
    
    if is_frozen:
        # Transfer learning mode - more lenient
        ratio = num_samples / trainable if trainable > 0 else float('inf')
        print(f"  Mode: Transfer Learning (frozen backbone)")
        print(f"  Samples per Trainable Parameter: {ratio:.2f}")
        
        if ratio >= 10:
            print(f"  Status: 🟢 EXCELLENT - Very low overfitting risk")
            print(f"  Recommendation: Train with confidence!")
        elif ratio >= 1:
            print(f"  Status: 🟡 GOOD - Manageable with regularization")
            print(f"  Recommendation: Use dropout=0.5, early stopping")
        elif ratio >= 0.1:
            print(f"  Status: 🟠 MODERATE - Use strong regularization")
            print(f"  Recommendation: dropout=0.5, weight_decay=1e-4, early stopping=5")
        else:
            print(f"  Status: 🔴 RISKY - Consider reducing model size")
            print(f"  Recommendation: Use HIDDEN_DIM=64 or fewer attributes")
    else:
        # Training from scratch - strict
        ratio = num_samples / trainable if trainable > 0 else float('inf')
        print(f"  Mode: Training from Scratch (unfrozen)")
        print(f"  Samples per Trainable Parameter: {ratio:.6f}")
        print(f"  Status: 🔴 DANGER - Severe overfitting guaranteed")
        print(f"  Recommendation: Enable FREEZE_BACKBONE=true")
    
    # Additional recommendations
    print(f"\nTraining Recommendations:")
    if is_frozen and trainable < 500000:
        print(f"  ✓ Batch Size: 32-64")
        print(f"  ✓ Learning Rate: 1e-3 (can be higher with frozen backbone)")
        print(f"  ✓ Weight Decay: 1e-4")
        print(f"  ✓ Dropout: 0.5")
        print(f"  ✓ Early Stopping: patience=10")
        print(f"  ✓ Expected Training Time: ~30-60 min on GPU")
    
    print("=" * 60)
    
    return model, total, trainable

# Test configurations
configs = {
    "full_unfrozen": {
        "IMAGE_BACKBONE": "resnet50",
        "TEXT_ENCODER": "bert-base-uncased",
        "HIDDEN_DIM": 256,
        "DROPOUT": 0.3,
        "FREEZE_BACKBONE": False,
        "ATTRIBUTES": {
            "sentiment": {"num_classes": 3},
            "emotion": {"num_classes": 6},
            "theme": {"num_classes": 8}
        }
    },
    "recommended_frozen": {
        "IMAGE_BACKBONE": "resnet18",
        "TEXT_ENCODER": "distilbert-base-uncased",
        "HIDDEN_DIM": 128,
        "DROPOUT": 0.5,
        "FREEZE_BACKBONE": True,
        "ATTRIBUTES": {
            "sentiment": {"num_classes": 3},
            "emotion": {"num_classes": 6},
            "theme": {"num_classes": 8}
        }
    }
}

if __name__ == "__main__":
    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"{'='*60}")
        try:
            analyze_model(cfg, num_samples=10000)
        except Exception as e:
            print(f"Error: {e}")