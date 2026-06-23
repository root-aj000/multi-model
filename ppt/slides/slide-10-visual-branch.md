# Slide 10 — FG_MFN Model (Visual Branch)

## What to Say (Speaker Notes)
"Let's zoom into the **visual branch** of the model. The visual branch is responsible for understanding the **image** — colors, shapes, objects, faces, layout. I use **ResNet-50**, which is a famous convolutional neural network (CNN) that has been **pre-trained on ImageNet** — a dataset of 1.2 million images. Pre-training means the network already knows how to see general things like edges, textures, and objects. I then **fine-tune** it on our ad dataset. Because our dataset is small (~4,860 images), I have an option to **freeze the backbone** — meaning the ResNet-50 weights don't change during training, and only the new layers I add on top get trained. This prevents overfitting. The output of ResNet-50 is a **2048-dimensional feature vector** for each image. The input image is resized to **224×224 pixels** with 3 color channels (RGB), and normalized using ImageNet statistics."

## What to Show on Screen

```
🖼️ VISUAL BRANCH — ResNet-50

   Input Image (224 × 224 × 3)
            │
            ▼
   ┌────────────────────┐
   │   ResNet-50        │
   │   (pretrained on   │
   │    ImageNet)       │
   │                    │
   │   50 layers deep   │
   └─────────┬──────────┘
             │
             ▼
   Visual Feature Vector (2048-d)

   ⚙️  KEY SETTINGS:
      • Pretrained weights: ImageNet
      • Optional FREEZE_BACKBONE=true
        (combat overfitting on small dataset)
      • Input: 224 × 224 × 3, ImageNet normalization
      • Output: 2048-d feature per image
```

## Visual Suggestion
- Show a **small picture of a ResNet-50 architecture** (the famous "identity shortcut" diagram).
- Highlight the **input** (224×224×3) and **output** (2048-d) clearly.
- Add a small note about **"frozen vs unfrozen"** with a snowflake icon.

## Key Talking Points
- ResNet-50 is a **battle-tested** architecture — used everywhere from medical imaging to self-driving cars.
- The **2048-d vector** is a compact summary of the image — every image becomes a point in 2048-dimensional space.
- **Freezing** is a common trick when you have little data.
