# Multi-Model (Multi-Modal) System — Deep Technical Reference

> **Complete internal architecture, tensor shapes, mathematical operations, parameter counts, training mechanics, and every design decision — explained from first principles.**

---

## Table of Contents

1. [Foundations](#1-foundations)
   - 1.1 [What is a Neural Network?](#11-what-is-a-neural-network)
   - 1.2 [What is a Feature Vector?](#12-what-is-a-feature-vector)
   - 1.3 [What is an Encoder?](#13-what-is-an-encoder)
   - 1.4 [What is "Multi-Modal"?](#14-what-is-multi-modal)
2. [The Problem](#2-the-problem)
3. [Architecture Overview](#3-architecture-overview)
4. [Component 1: OCR — Reading Text from Images](#4-component-1-ocr--reading-text-from-images)
   - 4.1 [How OCR Works](#41-how-ocr-works)
   - 4.2 [EasyOCR Engine](#42-easyocr-engine)
   - 4.3 [PaddleOCR Engine](#43-paddleocr-engine)
   - 4.4 [Factory Pattern](#44-factory-pattern)
   - 4.5 [Text Cleaning](#45-text-cleaning)
   - 4.6 [Tokenization](#46-tokenization)
5. [Component 2: Visual Encoder (ResNet-18)](#5-component-2-visual-encoder-resnet-18)
   - 5.1 [What is a CNN?](#51-what-is-a-cnn)
   - 5.2 [ResNet-18 Architecture](#52-resnet-18-architecture)
   - 5.3 [Stripping the Classification Head](#53-stripping-the-classification-head)
   - 5.4 [Freezing the Backbone](#54-freezing-the-backbone)
   - 5.5 [Supported Backbones](#55-supported-backbones)
   - 5.6 [Forward Pass — Every Tensor Shape](#56-forward-pass--every-tensor-shape)
6. [Component 3: Text Encoder (DistilBERT)](#6-component-3-text-encoder-distilbert)
   - 6.1 [What is a Transformer?](#61-what-is-a-transformer)
   - 6.2 [DistilBERT Architecture](#62-distilbert-architecture)
   - 6.3 [Tokenization Process](#63-tokenization-process)
   - 6.4 [Embedding Layer](#64-embedding-layer)
   - 6.5 [Self-Attention Mechanism](#65-self-attention-mechanism)
   - 6.6 [Transformer Block](#66-transformer-block)
   - 6.7 [Pooling Strategies](#67-pooling-strategies)
   - 6.8 [Attention-Weighted Pooling — Deep Dive](#68-attention-weighted-pooling--deep-dive)
   - 6.9 [Forward Pass — Every Tensor Shape](#69-forward-pass--every-tensor-shape)
7. [Component 4: Cross-Modal Attention Fusion](#7-component-4-cross-modal-attention-fusion)
   - 7.1 [Why Not Simple Concatenation?](#71-why-not-simple-concatenation)
   - 7.2 [What is Attention?](#72-what-is-attention)
   - 7.3 [Multi-Head Attention](#73-multi-head-attention)
   - 7.4 [Bidirectional Cross-Modal Attention](#74-bidirectional-cross-modal-attention)
   - 7.5 [Residual Connections and Layer Normalization](#75-residual-connections-and-layer-normalization)
   - 7.6 [Fusion Projection](#76-fusion-projection)
   - 7.7 [Forward Pass — Every Tensor Shape](#77-forward-pass--every-tensor-shape)
   - 7.8 [Gradient Flow](#78-gradient-flow)
   - 7.9 [Parameter Count](#79-parameter-count)
8. [Component 5: Shared Representation Layer](#8-component-5-shared-representation-layer)
   - 8.1 [Fully Connected Layers](#81-fully-connected-layers)
   - 8.2 [GELU Activation](#82-gelu-activation)
   - 8.3 [Dropout Regularization](#83-dropout-regularization)
   - 8.4 [Why Two Layers with Different Dropout?](#84-why-two-layers-with-different-dropout)
9. [Component 6: Classification Heads](#9-component-6-classification-heads)
   - 9.1 [Why Separate Heads?](#91-why-separate-heads)
   - 9.2 [Per-Head Architecture](#92-per-head-architecture)
   - 9.3 [Stop-Gradient Mechanism](#93-stop-gradient-mechanism)
   - 9.4 [The 9 Attributes](#94-the-9-attributes)
10. [Training Pipeline — Complete Mechanics](#10-training-pipeline--complete-mechanics)
    - 10.1 [Data Loading](#101-data-loading)
    - 10.2 [Image Preprocessing](#102-image-preprocessing)
    - 10.3 [Text Preprocessing](#103-text-preprocessing)
    - 10.4 [Label Encoding](#104-label-encoding)
    - 10.5 [DataLoader Configuration](#105-dataloader-configuration)
    - 10.6 [Augmentation](#106-augmentation)
    - 10.7 [Class Weights](#107-class-weights)
    - 10.8 [Optimizer — AdamW](#108-optimizer--adamw)
    - 10.9 [Parameter Grouping](#109-parameter-grouping)
    - 10.10 [Learning Rate Scheduler](#1010-learning-rate-scheduler)
    - 10.11 [Loss Function](#1011-loss-function)
    - 10.12 [Label Smoothing](#1012-label-smoothing)
    - 10.13 [Mixup Augmentation](#1013-mixup-augmentation)
    - 10.14 [Gradient Clipping](#1014-gradient-clipping)
    - 10.15 [Training Loop — Step by Step](#1015-training-loop--step-by-step)
    - 10.16 [Validation Loop](#1016-validation-loop)
    - 10.17 [Early Stopping](#1017-early-stopping)
11. [Inference Pipeline — End to End](#11-inference-pipeline--end-to-end)
    - 11.1 [Prediction Pipeline Construction](#111-prediction-pipeline-construction)
    - 11.2 [Single Image Prediction](#112-single-image-prediction)
    - 11.3 [Batch Prediction](#113-batch-prediction)
    - 11.4 [Post-Processing](#114-post-processing)
12. [Web Application](#12-web-application)
    - 12.1 [FastAPI Server](#121-fastapi-server)
    - 12.2 [API Endpoints](#122-api-endpoints)
    - 12.3 [Authentication (Supabase)](#123-authentication-supabase)
    - 12.4 [Request Flow](#124-request-flow)
    - 12.5 [Error Handling](#125-error-handling)
13. [Complete End-to-End Dimension Trace](#13-complete-end-to-end-dimension-trace)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Design Decisions — Why Each Choice](#15-design-decisions--why-each-choice)
16. [Terminology Glossary](#16-terminology-glossary)

---

## 1. Foundations

### 1.1 What is a Neural Network?

A neural network is a mathematical function composed of **layers**. Each layer performs two operations:

1. **Linear transformation**: `y = Wx + b` where `W` is a matrix of weights, `x` is the input, and `b` is a bias vector.
2. **Non-linear activation**: `y = activation(y)` — this allows the network to learn complex patterns, not just straight lines.

A **deep** neural network has many layers stacked together. The "deep" in "deep learning" refers to this depth.

**How it learns:**
1. Make a prediction (forward pass)
2. Measure how wrong the prediction is (loss)
3. Calculate which direction to adjust each weight (backward pass / gradients)
4. Adjust weights to reduce the loss (optimizer step)
5. Repeat thousands of times

**Weights** are the numbers inside the network. A network with 1 million parameters has 1 million numbers that get adjusted during training. The art of deep learning is designing architectures that have the right structure for the problem.

---

### 1.2 What is a Feature Vector?

A **feature vector** is a list of numbers that represents the "meaning" of some input.

**Example — an image:**
- Raw input: 512 x 512 x 3 = 786,432 pixel values (meaningless individually)
- Feature vector: 512 numbers that capture "what's in the image" (meaningful)

**Example — text:**
- Raw input: "Buy Now 50% Off Premium Headphones" (a string of characters)
- Feature vector: 768 numbers that capture "what the text means" (semantic encoding)

The magic of deep learning is that these feature vectors are **learned automatically**. The network discovers which features are important by looking at thousands of examples.

---

### 1.3 What is an Encoder?

An **encoder** is a neural network that converts raw data into a feature vector.

```
Raw Data → [Encoder Neural Network] → Feature Vector
```

Think of it like a summarizer:
- Input: a long article
- Encoder: reads and understands the article
- Output: a short summary (the feature vector)

The encoder compresses the input while preserving its essential meaning. Different encoders specialize in different types of data:
- **CNNs** (Convolutional Neural Networks) encode images
- **Transformers** encode text (and can also encode images)

---

### 1.4 What is "Multi-Modal"?

**Modal** means "type" or "form." Multi-modal means processing multiple types of data simultaneously.

This project processes **two modalities**:
1. **Visual modality** — the advertisement image (pixels)
2. **Textual modality** — the text extracted from the image (words)

**Why not just use one?**

An ad is more than its image or its text alone:
- A luxury watch ad has a shiny product image AND words like "premium," "exclusive," "limited edition"
- A fast-food ad has colorful food images AND "50% off," "buy now," "limited time"

Using only the image, you miss the textual urgency. Using only the text, you miss the visual appeal. Multi-modal fusion captures **both**.

---

## 2. The Problem

This system analyzes advertisements and makes **8 simultaneous predictions** about them:

| # | Attribute | Classes | Example |
|---|-----------|---------|---------|
| 1 | **theme** | 9 (Food, Fashion, Tech, Finance, Health, Gaming, Home, Education, Automotive) | What industry does the ad belong to? |
| 2 | **sentiment** | 3 (Positive, Negative, Neutral) | What overall emotional tone? |
| 3 | **emotion** | 5 (Anger, Excitement, Fear, Joy, Trust) | What specific feeling does it evoke? |
| 4 | **dominant_colour** | 10 (Red, Black, Blue, Green, White, Grey, Yellow, Brown, Orange, Purple) | What color dominates the ad? |
| 5 | **attention_score** | 3 (High, Medium, Low) | How attention-grabbing is it? |
| 6 | **trust_safety** | 3 (Safe, Unsafe, Questionable) | Is the ad trustworthy? |
| 7 | **predicted_ctr** | 3 (High, Medium, Low) | Will people click on it? |
| 8 | **likelihood_shares** | 3 (High, Medium, Low) | Will people share it? |

**Total: 39 output classes across 8 heads** (9+3+5+10+3+3+3+3 = 39)

---

## 3. Architecture Overview

The system is called **FG_MFN** (Fine-Grained Multi-Modal Fusion Network). Here is its complete structure:

```
INPUT: Ad Image (512x512 RGB) + Text (extracted via OCR)
                    |
        +-----------+-----------+
        |                       |
  [Visual Encoder]        [Text Encoder]
  ResNet-18 (frozen)      DistilBERT
  Image -> 512-d          Text -> 768-d
        |                       |
        +-----------+-----------+
                    |
          [Cross-Modal Attention]
          8-head bidirectional
          Visual attends to Text
          Text attends to Visual
          Output: 512-d
                    |
           [Shared FC Layer]
           Linear(512,512) -> GELU -> Dropout(0.5)
           Linear(512,512) -> GELU -> Dropout(0.25)
                    |
        +-----------+-----------+-----------+ ...
        |           |           |           |
    [Head 1]   [Head 2]   [Head 3]   [Head 4] ...
     theme     sentiment   emotion    color
     (9)       (3)         (5)        (10)
        |           |           |           |
     logits      logits      logits      logits
```

---

## 4. Component 1: OCR — Reading Text from Images

### 4.1 How OCR Works

**OCR (Optical Character Recognition)** converts text visible in images into machine-readable text.

**The process:**
1. **Text Detection**: Find rectangular regions in the image that contain text
2. **Character Recognition**: Classify each character region as a letter, number, or symbol
3. **Assembly**: Combine characters into words and sentences
4. **Confidence Scoring**: Assign a confidence score to each detection

**Why OCR is needed here:**
The neural network that processes text (DistilBERT) cannot read pixels. It needs actual text strings. OCR bridges the gap between the visual world (pixels containing text) and the textual world (string representations of that text).

---

### 4.2 EasyOCR Engine

**Class**: `EasyOCREngine` (`lib/ocr/easyocr.py`)

**Initialization:**
```python
easyocr.Reader(languages=["en"], model_storage_directory="local/ocr")
```
- Loads a CRAFT-based text detector + CRNN-based text recognizer
- Stores models in `local/ocr` directory
- Supports 80+ languages

**Text Extraction:**
```python
results = self.reader.readtext(image)
# results = [(bbox, text, confidence), ...]
```
- Input: numpy array (H, W, C) or file path
- Output: list of tuples — bounding box, detected text string, confidence score
- The engine averages confidence across all detections: `sum(confidences) / len(confidences)`
- Returns `(combined_text, average_confidence)`

**Error Handling:**
- If reader fails to initialize: `self.reader = None`, raises `RuntimeError` on extraction
- If no text detected: returns `("", 0.0)`

---

### 4.3 PaddleOCR Engine

**Class**: `PaddleOCREngine` (`lib/ocr/paddleocr.py`)

**Initialization:**
```python
PaddleOCR(use_angle_cls=True, lang="en", det_model_dir="local/ocr")
```
- `use_angle_cls=True`: enables text angle classification (handles rotated text)
- PaddleOCR uses a three-stage pipeline: detection → classification → recognition

**Text Extraction:**
```python
result = self.ocr.ocr(image, cls=True)
# result = [[(coords, (text, confidence)), ...]]
```
- Validates result structure: checks `result`, `result[0]`, and list type
- Extracts text and confidence from nested list structure
- Returns `("", 0.0)` if no text detected

**Difference from EasyOCR:**
- EasyOCR expects `languages` as a list: `["en"]`
- PaddleOCR expects `language` as a string: `"en"`
- PaddleOCR uses PaddlePaddle framework; EasyOCR uses PyTorch

---

### 4.4 Factory Pattern

**Function**: `create_ocr_engine(engine_name, model_dir, **kwargs)` (`lib/ocr/factory.py`)

Uses the **Factory Pattern** — a design pattern that creates objects without specifying the exact class:

```python
SUPPORTED_OCR_ENGINES = ["easyocr", "paddleocr"]

if engine_name == "easyocr":
    return EasyOCREngine(model_dir, languages=kwargs.get("languages", ["en"]))
elif engine_name == "paddleocr":
    return PaddleOCREngine(model_dir, language=kwargs.get("language", "en"))
else:
    raise ValueError(f"Unsupported engine: {engine_name}")
```

**Why a factory?**
- The prediction pipeline doesn't need to know which OCR engine is being used
- Adding a new engine (e.g., Tesseract) requires only adding a new class and a new `elif` branch
- The rest of the code stays unchanged

---

### 4.5 Text Cleaning

After OCR extracts raw text, it is cleaned by `clean_text()` (`lib/preprocessing/text/cleaner.py`).

**The cleaning pipeline:**

```
Raw OCR text: "B¥ Now! 50% OffPremium  Headphones\x00\x01"
    |
    v
1. lowercase():      "b¥ now! 50% offpremium  headphones\x00\x01"
    |
    v
2. strip():          "b¥ now! 50% offpremium  headphones\x00\x01"
    |
    v
3. regex remove      "b now 50 offpremium headphones"
   [^a-z0-9\s$€£₹¥%.,!?\-&/@]
   (keep only alphanumeric, whitespace, currency, percent, basic punct)
    |
    v
4. collapse whitespace: "b now 50 offpremium headphones"
    |
    v
5. strip():          "b now 50 offpremium headphones"
```

**Why this aggressive cleaning?**
- OCR produces artifacts: `¥` instead of `Y`, random symbols, control characters
- The tokenizer (DistilBERT) was trained on clean English text
- Feeding noisy OCR text would confuse the encoder
- The regex whitelist `[a-z0-9\s$€£₹¥%.,!?-&/@]` keeps meaningful content (prices, percentages) while removing noise

**For training**, a lighter cleaner is used (`clean_adcopy`):
- Keeps apostrophes, punctuation, standard text
- Only removes control characters (`\x00-\x08`, `\x0B-\x0C`, `\x0E-\x1F`, `\x7F`)
- Normalizes Unicode to NFC form (e.g., `é` as single codepoint instead of `e` + accent)

---

### 4.6 Tokenization

**Function**: `tokenize_text(text, max_length=256)` (`lib/preprocessing/text/tokenizer.py`)

**Why tokenization is needed:**
Neural networks cannot process strings. They process numbers. Tokenization converts text into sequences of integer IDs.

**Step-by-step process:**

```
Input: "buy now 50 off premium headphones"
    |
    v
1. WordPiece tokenization:
   ["[CLS]", "buy", "now", "50", "off", "prem", "##ium", "head", "##phones", "[SEP]", "[PAD]", ...]
   - Each token is a subword unit
   - "[CLS]" = special classification token (prepended)
   - "[SEP]" = separator token (appended)
   - "[PAD]" = padding tokens (to reach max_length)
   - "##" prefix = continuation of previous word
    |
    v
2. Convert to integer IDs:
   [101, 4897, 2085, 2423, 2125, 2148, 11336, 2023, 4674, 102, 0, 0, ...]
   - Each token maps to an integer in the vocabulary
   - Vocabulary size: 30,522 tokens
    |
    v
3. Create attention mask:
   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
   - 1 = real token, 0 = padding token
   - Tells the model to ignore padding positions
    |
    v
4. Truncate to max_length (256):
   Both input_ids and attention_mask are exactly 256 elements
```

**Empty text handling:**
If the OCR returns empty text, the tokenizer returns all-zero tensors:
```python
input_ids = torch.zeros(max_length, dtype=torch.long)       # [0, 0, 0, ...]
attention_mask = torch.zeros(max_length, dtype=torch.long)   # [0, 0, 0, ...]
```
This prevents crashes when ads contain no text.

---

## 5. Component 2: Visual Encoder (ResNet-18)

### 5.1 What is a CNN?

A **Convolutional Neural Network (CNN)** is a neural network designed for grid-like data (images). It uses **convolution** — sliding a small filter over the image to detect patterns.

**How convolution works:**

```
Image patch:          Filter (3x3):
[120, 130, 125]      [1, 0, -1]
[115, 128, 132]  *   [1, 0, -1]
[122, 127, 130]      [1, 0, -1]
= 120*1 + 130*0 + 125*(-1) + 115*1 + 128*0 + 132*(-1) + 122*1 + 127*0 + 130*(-1)
= 120 - 125 + 115 - 132 + 122 - 130
= -30
```

The filter detects **vertical edges** (large difference between left and right). Different filters detect different patterns:
- Horizontal edges
- Color gradients
- Textures
- Shapes

**CNN layers learn hierarchically:**
- Early layers: edges, colors, textures (simple patterns)
- Middle layers: shapes, parts of objects (complex patterns)
- Late layers: whole objects, faces, scenes (semantic understanding)

---

### 5.2 ResNet-18 Architecture

**ResNet** (Residual Network) introduced **skip connections** — connections that jump over one or more layers:

```
Input → Conv → BN → ReLU → Conv → BN → (+ Input) → ReLU
         \__________________________________________/
                     skip connection
```

**Why skip connections?**
Without them, gradients (the signals that update weights) get smaller and smaller as they flow backward through many layers (the "vanishing gradient" problem). Skip connections provide a direct path for gradients.

**ResNet-18 layers:**

```
Input: (B, 3, 512, 512) — batch of 3-channel 512x512 images

conv1:    Conv2d(3, 64, kernel=7, stride=2, padding=3) → (B, 64, 256, 256)
bn1:      BatchNorm2d(64)
relu1:    ReLU
maxpool:  MaxPool2d(3, stride=2, padding=1) → (B, 64, 128, 128)

layer1:   2x BasicBlock(64, 64)     → (B, 64, 128, 128)
layer2:   2x BasicBlock(64, 128)    → (B, 128, 64, 64)     (stride 2)
layer3:   2x BasicBlock(128, 256)   → (B, 256, 32, 32)     (stride 2)
layer4:   2x BasicBlock(256, 512)   → (B, 512, 16, 16)     (stride 2)

avgpool:  AdaptiveAvgPool2d(1, 1)   → (B, 512, 1, 1)
flatten:  → (B, 512)
```

**Each BasicBlock:**
```
Input → Conv(3x3) → BN → ReLU → Conv(3x3) → BN → (+ Input) → ReLU
```

**Total parameters:** ~11.2 million (ResNet-18)

---

### 5.3 Stripping the Classification Head

ResNet-18 was originally trained on ImageNet (1,000 classes). Its final layer is:
```python
self.fc = nn.Linear(512, 1000)  # 512 inputs -> 1000 class scores
```

We don't want 1,000 ImageNet class scores. We want the **512-dimensional feature vector** that comes before this layer. So we replace `fc` with `nn.Identity()` (a pass-through):

```python
model.fc = nn.Identity()  # Now the model outputs 512-d features instead of 1000-d class scores
```

For EfficientNet/ConvNeXt models, the head is `nn.Sequential(Dropout, Linear)`. We replace the last element (the Linear layer) with `nn.Identity()`.

---

### 5.4 Freezing the Backbone

**What freezing means:**
Setting `param.requires_grad = False` for all backbone parameters. This means:
- The backbone still computes features (forward pass works)
- But gradients are NOT computed for backbone weights (saves memory)
- And backbone weights are NOT updated by the optimizer

**Why freeze?**

The backbone (ResNet-18) has **11.2 million parameters**. The entire dataset has only **4,860 images**. If all 11.2M parameters are trainable:
- The model can memorize every training image (overfitting)
- It won't generalize to new images
- Training will be slow (11.2M gradients per batch)

By freezing:
- The backbone acts as a **fixed feature extractor** (like a pre-built function)
- Only the text encoder (~66M params), fusion layer (~6.5M), shared FC (~525K), and heads (~1M) are trainable
- The pretrained ImageNet knowledge is preserved

**What gets frozen vs. trainable:**

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| ResNet-18 backbone | ~11.2M | No (frozen) |
| DistilBERT encoder | ~66.4M | Yes |
| CrossModalAttention | ~6.5M | Yes |
| Shared FC | ~525K | Yes |
| 8 Classification heads | ~1.1M | Yes |
| **Total** | **~85.6M** | **~74.4M trainable** |

---

### 5.5 Supported Backbones

All backbones are registered in `_BACKBONE_REGISTRY`:

| Name | Constructor | Native Dim | Parameters | Notes |
|------|------------|------------|------------|-------|
| `resnet18` | `torchvision.models.resnet18` | 512 | ~11.2M | Default. Lightweight. |
| `resnet50` | `torchvision.models.resnet50` | 2048 | ~25.6M | More powerful, 4x more params. |
| `efficientnet_b0` | `torchvision.models.efficientnet_b0` | 1280 | ~5.3M | Best accuracy/parameter ratio. |
| `efficientnet_b3` | `torchvision.models.efficientnet_b3` | 1536 | ~12.2M | Better accuracy than B0. |
| `efficientnet_b7` | `torchvision.models.efficientnet_b7` | 2560 | ~66.3M | Very large, needs more data. |
| `convnext_tiny` | `torchvision.models.convnext_tiny` | 768 | ~5.8M | Modern ConvNet architecture. |
| `convnext_small` | `torchvision.models.convnext_small` | 768 | ~5.8M | Same as tiny, different depth. |
| `convnext_base` | `torchvision.models.convnext_base` | 1024 | ~89M | Largest option. |

**Why ResNet-18 by default?**
- Small dataset (4,860 images) → small model to prevent overfitting
- Fast inference (important for real-time predictions)
- Well-understood, battle-tested architecture
- 512-d output is manageable for downstream fusion

---

### 5.6 Forward Pass — Every Tensor Shape

```
INPUT: images (B, 3, 512, 512)
  |
  v
conv1: Conv2d(3, 64, kernel=7, stride=2, padding=3)
  weights: (64, 3, 7, 7) = 9,408 params
  output: (B, 64, 256, 256)
  |
  v
bn1: BatchNorm2d(64)
  weights: (64,) gamma + (64,) beta = 128 params
  output: (B, 64, 256, 256)
  |
  v
relu1: ReLU
  output: (B, 64, 256, 256) — negative values set to 0
  |
  v
maxpool: MaxPool2d(3, stride=2, padding=1)
  output: (B, 64, 128, 128)
  |
  v
layer1: 2x BasicBlock(64→64)
  Block 1: Conv(64,64,3) → BN → ReLU → Conv(64,64,3) → BN → (+input) → ReLU
  Block 2: same
  output: (B, 64, 128, 128)
  |
  v
layer2: 2x BasicBlock(64→128) — first block stride=2
  Block 1: Conv(64,128,3,stride=2) → BN → ReLU → Conv(128,128,3) → BN → (+proj) → ReLU
  Block 2: Conv(128,128,3) → BN → ReLU → Conv(128,128,3) → BN → (+input) → ReLU
  output: (B, 128, 64, 64)
  |
  v
layer3: 2x BasicBlock(128→256) — first block stride=2
  output: (B, 256, 32, 32)
  |
  v
layer4: 2x BasicBlock(256→512) — first block stride=2
  output: (B, 512, 16, 16)
  |
  v
avgpool: AdaptiveAvgPool2d(1, 1)
  output: (B, 512, 1, 1) — each 16x16 spatial feature map averaged to single value
  |
  v
flatten: (B, 512, 1, 1) → (B, 512)
  |
  v
Identity() (stripped fc): no change
  |
  v
OUTPUT: visual_features (B, 512)
```

---

## 6. Component 3: Text Encoder (DistilBERT)

### 6.1 What is a Transformer?

A **Transformer** is a neural network architecture that processes sequences (like text) using **self-attention**. Unlike older architectures (RNNs, LSTMs) that process one word at a time left-to-right, Transformers process ALL words simultaneously.

**Key insight: self-attention.**

For each word, self-attention asks: "How much should I attend to every other word?"

```
Sentence: "The premium headphones are on sale"

For "premium":
  - attends heavily to "headphones" (it describes what's premium)
  - attends to "sale" (it's the context)
  - attends little to "The" (function word, less important)

For "sale":
  - attends heavily to "on" (on sale = phrase)
  - attends to "premium headphones" (the product being sold)
```

This allows the model to understand context and relationships between any pair of words, regardless of distance.

---

### 6.2 DistilBERT Architecture

**DistilBERT** is a distilled (compressed) version of BERT:

| Property | BERT-base | DistilBERT |
|----------|-----------|------------|
| Layers | 12 | 6 |
| Hidden size | 768 | 768 |
| Attention heads | 12 | 12 |
| Parameters | 110M | 66M |
| Speed | 1x | 1.6x faster |
| Accuracy | 100% | 97% of BERT |

**Distillation** means training a smaller "student" model to mimic a larger "teacher" model. The student learns to produce similar outputs with fewer parameters.

**Architecture:**
```
Input tokens → Embedding → 6 Transformer Layers → Last Hidden State (B, seq_len, 768)
                                                         |
                                                    Pooling
                                                         |
                                                    Output (B, 768)
```

---

### 6.3 Tokenization Process

DistilBERT uses **WordPiece tokenization**:

**Vocabulary:** 30,522 tokens

**How WordPiece works:**
1. Start with a pre-built vocabulary of common words and subwords
2. For each word, try to match it to vocabulary entries
3. If no exact match, split into subwords with `##` prefix

**Examples:**
```
"headphones" → ["head", "##phones"]     (head is common, phones needs ##)
"unhappiness" → ["un", "##happy", "##ness"]  (three subwords)
"buy" → ["buy"]                          (exact match)
"50" → ["50"]                            (exact match)
```

**Special tokens:**
- `[CLS]` (token ID 101): prepended to every sequence. Its final hidden state is used as the "sentence representation" in CLS pooling.
- `[SEP]` (token ID 102): marks end of sequence.
- `[PAD]` (token ID 0): padding to reach `max_length=256`.

**Padding and truncation:**
- Texts shorter than 256 tokens: padded with `[PAD]` tokens
- Texts longer than 256 tokens: truncated (first 256 tokens kept)
- Attention mask: 1 for real tokens, 0 for padding

---

### 6.4 Embedding Layer

The embedding layer converts integer token IDs into dense vectors:

```python
Embedding(30522, 768)  # 30,522 vocabulary entries, each mapped to 768-d vector
```

**Two embeddings are summed:**
1. **Token embedding**: looks up the vector for each token
2. **Position embedding**: adds position information (Transformers have no inherent sense of order)

```
token_ids: [101, 4897, 2085, 2423, ...]
    ↓
token_embeddings: (B, seq_len, 768)  — lookup from embedding table
position_embeddings: (B, seq_len, 768) — learned position vectors
    ↓
embedded = token_embeddings + position_embeddings
    ↓
LayerNorm + Dropout
    ↓
output: (B, seq_len, 768)
```

**Parameters:** 30,522 × 768 = 23,440,896 (token embeddings) + 256 × 768 = 196,608 (position embeddings)

---

### 6.5 Self-Attention Mechanism

**Scaled Dot-Product Attention:**

For each attention head, with dimension `d_k = 768/12 = 64`:

```
Q = X @ W_Q    (B, seq_len, 768) @ (768, 64) = (B, seq_len, 64)
K = X @ W_K    (B, seq_len, 768) @ (768, 64) = (B, seq_len, 64)
V = X @ W_V    (B, seq_len, 768) @ (768, 64) = (B, seq_len, 64)

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Q @ K^T:      (B, seq_len, 64) @ (B, 64, seq_len) = (B, seq_len, seq_len)
              — each element [i,j] = how much token i attends to token j

/sqrt(64):    scale by sqrt(d_k) = 8 — prevents dot products from growing too large
              (large values push softmax to one-hot, killing gradients)

softmax:      (B, seq_len, seq_len) — each row sums to 1 (probability distribution)

@ V:          (B, seq_len, seq_len) @ (B, seq_len, 64) = (B, seq_len, 64)
              — weighted sum of value vectors, weighted by attention scores
```

**Multi-head attention (12 heads):**
Each head has its own W_Q, W_K, W_V (64-d each). Results are concatenated:

```
head_1: (B, seq_len, 64)
head_2: (B, seq_len, 64)
...
head_12: (B, seq_len, 64)
    ↓ concatenate
(B, seq_len, 768)
    ↓ output projection
Linear(768, 768) → (B, seq_len, 768)
```

---

### 6.6 Transformer Block

Each of DistilBERT's 6 layers contains:

```
Input: (B, seq_len, 768)
    |
    v
LayerNorm
    |
    v
Multi-Head Self-Attention (12 heads, 64-d each)
    |
    v
Dropout
    |
    v
(+ Input) — residual connection
    |
    v
LayerNorm
    |
    v
Feed-Forward Network:
    Linear(768, 3072) → GELU → Linear(3072, 768)
    (expand 4x, activate, project back)
    |
    v
Dropout
    |
    v
(+ Previous) — residual connection
    |
    v
Output: (B, seq_len, 768)
```

**Parameters per layer:**
- Self-attention: 4 × (768 × 64 × 12) = 2,359,296 (Q,K,V projections + output)
- FFN: (768 × 3072) + (3072 × 768) = 4,718,592
- Layer norms: 2 × (768 × 2) = 3,072
- **Total per layer:** ~7.1M
- **Total across 6 layers:** ~42.6M

---

### 6.7 Pooling Strategies

After the 6 transformer layers, we have `(B, seq_len, 768)` — a vector for every token. We need a single `(B, 768)` vector for the entire text. This is called **pooling**.

**Three strategies:**

**1. CLS Pooling:**
```python
pooled = last_hidden[:, 0, :]  # Take the [CLS] token's output
```
- Simple: just grab the first token
- The [CLS] token is trained to aggregate sentence-level information
- Downside: may not capture all nuance

**2. Mean Pooling:**
```python
mask = attention_mask.unsqueeze(-1).float()      # (B, seq_len, 1)
pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```
- Averages all non-padding token vectors
- More representative than CLS
- Downside: treats all tokens equally (keywords and fillers weighted the same)

**3. Attention-Weighted Pooling (default):**
```python
# Uses the model's own attention scores as importance weights
```
This is the best strategy — see next section.

---

### 6.8 Attention-Weighted Pooling — Deep Dive

**The key idea:** Use the Transformer's own learned attention scores to weight token importance.

**Step-by-step math:**

```
Step 1: Get last layer attention weights
  attentions[-1]: (B, 12, seq_len, seq_len)
  — 12 attention heads, each with a (seq_len x seq_len) attention matrix

Step 2: Average across heads
  avg_attn = attentions[-1].mean(dim=1)   # (B, seq_len, seq_len)
  avg_attn[b, i, j] = average attention from token i to token j

Step 3: Compute per-token importance
  token_importance = avg_attn.sum(dim=1)   # (B, seq_len)
  token_importance[b, j] = sum over all i of avg_attn[b, i, j]
  — "How much attention does token j receive from ALL other tokens?"

Step 4: Mask padding
  token_importance = token_importance * attention_mask.float()
  — Padding tokens get 0 importance

Step 5: Normalize to probability distribution
  token_importance = token_importance / token_importance.sum(dim=1, keepdim=True).clamp(min=1e-9)
  — Now sums to 1.0 per sample

Step 6: Weighted sum
  pooled = (token_importance.unsqueeze(-1) * last_hidden).sum(dim=1)   # (B, 768)
  — Each token's 768-d vector is weighted by its importance score
```

**Why this is superior:**
- Keywords ("premium," "50% off," "buy now") naturally receive more attention → higher weight
- Fillers ("the," "a," "is") receive less attention → lower weight
- No extra parameters needed — uses the model's own learned attention
- Padding tokens are automatically zero-weighted

---

### 6.9 Forward Pass — Every Tensor Shape

```
INPUT: input_ids (B, 256), attention_mask (B, 256)
  |
  v
Embedding(30522, 768):
  token_emb: (B, 256, 768)
  pos_emb: (B, 256, 768) — learned position 0..255
  embedded = token_emb + pos_emb → (B, 256, 768)
  LayerNorm + Dropout → (B, 256, 768)
  |
  v
Transformer Layer 0:
  Self-Attention:
    Q = input @ W_Q: (B, 256, 768) @ (768, 64×12) = (B, 256, 768)
    K = input @ W_K: same
    V = input @ W_V: same
    → 12 heads of (B, 256, 64) → concat → (B, 256, 768) → output proj → (B, 256, 768)
  FFN:
    Linear(768, 3072): (B, 256, 3072)
    GELU: (B, 256, 3072)
    Linear(3072, 768): (B, 256, 768)
  Output: (B, 256, 768)
  |
  v
Transformer Layers 1-5: same structure
  Output: (B, 256, 768)
  |
  v
Pooling (attention-weighted):
  attentions[-1]: (B, 12, 256, 256)
  avg_attn: (B, 256, 256)
  token_importance: (B, 256)
  masked: (B, 256)
  normalized: (B, 256)
  weighted_sum: (B, 768)
  |
  v
OUTPUT: text_features (B, 768)
```

---

## 7. Component 4: Cross-Modal Attention Fusion

### 7.1 Why Not Simple Concatenation?

The naive approach to combining two feature vectors:

```python
fused = torch.cat([visual_features, text_features], dim=1)  # (B, 512+768) = (B, 1280)
```

**Problem:** This treats all features equally and independently. It doesn't learn which visual features relate to which text features.

**Example:**
- Ad shows a red sale tag + text says "50% off"
- Concatenation: [red_features, sale_features, 50_off_features, ...] — all mixed together
- The model must learn on its own that red + sale + 50% off are related
- This is possible but requires more data and deeper layers

**Cross-modal attention** explicitly models the relationships:
- "Which text tokens are relevant to this visual feature?"
- "Which visual features support this text token?"

---

### 7.2 What is Attention?

Attention is a mechanism that computes **relevance scores** between elements.

**General formula:**
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **Q @ K^T**: similarity between query and key (dot product)
- **softmax**: converts to probability distribution (sums to 1)
- **@ V**: weighted sum of values, weighted by relevance

---

### 7.3 Multi-Head Attention

Instead of computing one attention function, we compute **8 parallel attention functions** (heads):

```python
nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.5, batch_first=True)
```

**Each head:**
- Gets `768/8 = 96` dimensional Q, K, V
- Computes its own attention
- Learns different relationships

**Inside each head:**
```
Q: (B, 1, 768) → (B, 1, 96)  — per-head subspace
K: (B, 1, 768) → (B, 1, 96)
V: (B, 1, 768) → (B, 1, 96)

Attention scores: Q @ K^T / sqrt(96) = (B, 1, 1) — scalar (since seq_len=1)
softmax: (B, 1, 1) — just 1.0 (only one query-key pair)
@ V: (B, 1, 96)
```

**Concatenation across heads:**
```
head_0: (B, 1, 96)
head_1: (B, 1, 96)
...
head_7: (B, 1, 96)
    ↓ concatenate
(B, 1, 768)
    ↓ output projection
Linear(768, 768) → (B, 1, 768)
```

**Why 8 heads?**
- Original Transformer paper used 8 heads (Vaswani et al., 2017)
- Too few heads → miss diverse relationships
- Too many heads → each head gets too little information
- 8 is empirically well-balanced for 768-d embeddings

---

### 7.4 Bidirectional Cross-Modal Attention

The system computes attention in **two directions**:

**Direction 1: Visual attending to Text**
```
Query: visual features (projected to 768-d)
Key/Value: text features (projected to 768-d)

Meaning: "For each visual element, which text tokens are relevant?"
Output: visual features enriched with textual context
```

**Direction 2: Text attending to Visual**
```
Query: text features (768-d)
Key/Value: visual features (projected to 768-d)

Meaning: "For each text token, which visual features support it?"
Output: text features enriched with visual context
```

**Why bidirectional?**
- Visual → Text: "This red color relates to the word 'sale'"
- Text → Visual: "The word 'premium' relates to the shiny product in the image"
- Both directions provide complementary information

---

### 7.5 Residual Connections and Layer Normalization

After each attention computation, two operations stabilize training:

**Residual connection:**
```python
v_out = v + v_attended  # Add original input back
```
- Prevents information loss (if attention computes near-zero, the original is preserved)
- Provides a direct gradient path (gradients can flow through the addition without going through attention)

**Layer Normalization:**
```python
v_out = LayerNorm(v_out)
```
- Normalizes the 768 features to have mean=0, variance=1, then applies learned scale/shift
- Formula: `gamma * (x - mean) / sqrt(var + eps) + beta`
- Prevents values from exploding or vanishing
- `eps = 1e-5` (numerical stability)

---

### 7.6 Fusion Projection

After bidirectional attention, the two enriched representations are combined:

```python
fused = torch.cat([v_out, t_out], dim=1)   # (B, 768) + (B, 768) = (B, 1536)
fused = fusion_proj(fused)                   # Linear(1536, 512) → (B, 512)
```

**Why project to 512?**
- The visual encoder outputs 512-d
- The text encoder outputs 768-d
- Fusion produces 1536-d (concatenation of both)
- Projecting to 512-d:
  - Reduces dimensionality for efficiency
  - Matches the hidden dimension of the shared FC layer
  - Forces the model to learn a compressed, combined representation

---

### 7.7 Forward Pass — Every Tensor Shape

```
INPUT: visual (B, 512), text (B, 768)
  |
  v
visual_proj: Linear(512, 768)
  weights: (768, 512) = 393,216 params
  output: (B, 768)
  unsqueeze(1): (B, 1, 768)  — add seq_len=1 for MHA
  |
  v
text_proj: Linear(768, 768)
  weights: (768, 768) = 589,824 params
  output: (B, 768)
  unsqueeze(1): (B, 1, 768)
  |
  v
v2t_attn: MultiheadAttention(768, 8, dropout=0.5)
  in_proj_weight: (3×768, 768) = 1,769,472 params — Q,K,V projections
  in_proj_bias: (3×768,) = 2,304 params
  out_proj: (768, 768) = 590,592 params + bias 768
  Total: 2,363,136 params
  
  query = v: (B, 1, 768)
  key = t: (B, 1, 768)
  value = t: (B, 1, 768)
  
  Attention computation:
    Q = query @ in_proj_weight[:768] + bias[:768]: (B, 1, 768)
    K = query @ in_proj_weight[768:1536] + bias[768:1536]: (B, 1, 768)
    V = value @ in_proj_weight[1536:] + bias[1536:]: (B, 1, 768)
    
    Split into 8 heads:
      Q: (B, 8, 1, 96)
      K: (B, 8, 1, 96)
      V: (B, 8, 1, 96)
    
    Scaled dot-product:
      scores = Q @ K^T / sqrt(96): (B, 8, 1, 1) — scalar per head
      attn_weights = softmax(scores): (B, 8, 1, 1) — always 1.0 (single token)
      attn_output = attn_weights @ V: (B, 8, 1, 96)
    
    Concatenate heads:
      (B, 8, 1, 96) → (B, 1, 768)
    
    Output projection:
      Linear(768, 768): (B, 1, 768)
  
  dropout(0.5): (B, 1, 768) — training only
  |
  v
t2v_attn: MultiheadAttention(768, 8, dropout=0.5)
  Same computation but reversed:
    query = t, key = v, value = v
  output: (B, 1, 768)
  dropout(0.5): (B, 1, 768)
  |
  v
Residual + LayerNorm:
  v + v_attended: (B, 1, 768) + (B, 1, 768) = (B, 1, 768)
  v_norm: LayerNorm(768) → (B, 1, 768)
  squeeze(1): (B, 768)
  
  t + t_attended: (B, 1, 768) + (B, 1, 768) = (B, 1, 768)
  t_norm: LayerNorm(768) → (B, 1, 768)
  squeeze(1): (B, 768)
  |
  v
Concatenate:
  torch.cat([v_out, t_out], dim=1): (B, 768) + (B, 768) = (B, 1536)
  |
  v
fusion_proj: Linear(1536, 512)
  weights: (512, 1536) = 786,432 params
  output: (B, 512)
  |
  v
OUTPUT: fused (B, 512)
```

---

### 7.8 Gradient Flow

During backpropagation, gradients flow from the loss through every component:

```
Loss
  → fusion_proj gradients (Linear 1536→512)
  → split into v_out and t_out paths

  v_out path:
    → v_norm (gamma, beta gradients)
    → residual: gradient flows through BOTH v (original) and v_attended
      → v direct path: visual_proj gradients (BUT backbone is frozen, so stops)
      → v_attended path: v2t_attn gradients
        → output projection → attention weights → Q,K,V projections
        → visual_proj receives gradient (K/V came from text_proj)
    
  t_out path:
    → t_norm (gamma, beta gradients)
    → residual: gradient flows through BOTH t and t_attended
      → t direct path: text_proj gradients → DistilBERT encoder gradients
      → t_attended path: t2v_attn gradients
        → output projection → attention weights → Q,K,V projections
        → text_proj receives gradient (Q came from text_proj)
        → visual_proj receives gradient (K/V from visual_proj)

Total gradient paths from loss to encoder:
  1. text_proj → DistilBERT (full encoder backprop)
  2. visual_proj → but stops (frozen backbone)
  3. CrossModalAttention internal (fusion_proj, v2t_attn, t2v_attn, norms)
```

---

### 7.9 Parameter Count

| Module | Parameters |
|--------|-----------|
| visual_proj: Linear(512, 768) | 394,752 |
| text_proj: Linear(768, 768) | 590,592 |
| v2t_attn: MultiheadAttention(768, 8) | 2,363,136 |
| t2v_attn: MultiheadAttention(768, 8) | 2,363,136 |
| v_norm: LayerNorm(768) | 1,536 |
| t_norm: LayerNorm(768) | 1,536 |
| fusion_proj: Linear(1536, 512) | 786,944 |
| **Total** | **~6,501,632** |

---

## 8. Component 5: Shared Representation Layer

### 8.1 Fully Connected Layers

A **fully connected (FC) layer** connects every input neuron to every output neuron:

```
output_j = sum_i(input_i * weight_ij) + bias_j
```

For `Linear(512, 512)`:
- Weight matrix: (512, 512) = 262,144 parameters
- Bias vector: 512 parameters
- Total: 262,656 parameters

---

### 8.2 GELU Activation

**GELU = Gaussian Error Linear Unit**

```
GELU(x) = x * Phi(x)
```

Where `Phi(x)` is the CDF of the standard Gaussian distribution.

**Approximation:**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

**Why GELU instead of ReLU?**
- ReLU: `f(x) = max(0, x)` — hard threshold at 0
  - Problem: "dying neurons" — if a neuron's input is always negative, it outputs 0, gets 0 gradient, never learns
- GELU: smooth curve that gradually gates based on input magnitude
  - Near zero: small negative values still pass through (with small negative weight)
  - No dying neuron problem
  - Standard in Transformer architectures

---

### 8.3 Dropout Regularization

**Dropout** randomly sets a fraction of activations to zero during training:

```python
nn.Dropout(p=0.5)  # 50% of neurons randomly zeroed each forward pass
```

**How it works:**
```
Input: [0.2, -0.5, 0.8, 1.2, -0.1, 0.6]
  ↓ Dropout(0.5)
Random mask: [1, 0, 1, 0, 1, 0] (50% ones, 50% zeros)
Output: [0.2, 0, 0.8, 0, -0.1, 0] * (1/(1-0.5)) = [0.4, 0, 1.6, 0, -0.2, 0]
```

**Why the scaling factor `1/(1-p)`?**
Without scaling, the expected sum of activations would decrease during training. The scaling factor keeps the expected value the same as without dropout.

**Why dropout?**
- Prevents co-adaptation: neurons can't rely on specific other neurons always being present
- Forces redundancy: the network must learn multiple independent representations
- Acts as implicit ensemble: training many sub-networks, averaging at inference

---

### 8.4 Why Two Layers with Different Dropout?

```python
Shared FC (DEEP_SHARED_LAYER=True):
  Linear(512, 512)   → fusion_dim to hidden_dim
  GELU
  Dropout(0.5)        — strong regularization (sees raw fused features)
  Linear(512, 512)   → hidden_dim to hidden_dim
  GELU
  Dropout(0.25)       — lighter regularization (closer to classification)
```

**Why different dropout rates?**
- **First layer (0.5):** Receives raw fused features directly from cross-modal attention. These features may contain noise from the fusion process. Stronger dropout forces the network to be robust to this noise.
- **Second layer (0.25):** Receives already-processed features that are closer to the final representation. Less dropout preserves more information for classification.

**Why deep (2 layers) instead of shallow (1 layer)?**
- Deep: `Linear → GELU → Dropout → Linear → GELU → Dropout` — learns a transformation, then refines it
- Shallow: `Linear → GELU → Dropout` — single transformation
- Deep allows learning more complex feature interactions before classification

---

## 9. Component 6: Classification Heads

### 9.1 Why Separate Heads?

Each attribute has different characteristics:
- Different number of classes (3 vs 5 vs 9 vs 10)
- Different semantics (sentiment vs color vs engagement)
- Different difficulty levels

A single shared head would force all attributes into the same architecture, which is suboptimal. Separate heads allow each attribute to have its own specialized classifier.

---

### 9.2 Per-Head Architecture

Each head follows the same template:

```python
head = nn.Sequential(
    nn.Linear(512, 256),    # 512 input features → 256 hidden
    nn.GELU(),               # Non-linearity
    nn.Dropout(0.5),         # Regularization
    nn.Linear(256, num_classes)  # 256 hidden → num_classes outputs
)
```

**Hidden dimension formula:**
```python
head_hidden = max(hidden_dim // 2, num_classes * 4)
# = max(256, num_classes * 4)
```

For all attributes, `num_classes * 4 < 256`, so all heads use 256 hidden units.

**Parameters per head (e.g., 3-class head):**
- Linear(512, 256): 512 × 256 + 256 = 131,328
- Linear(256, 3): 256 × 3 + 3 = 771
- **Total:** ~132,099

**Total across all 8 heads:** ~1,056,792

---

### 9.3 Stop-Gradient Mechanism

Some attributes are noisy (CTR, shares have Cramer's V ~ 0.02 with content). Their gradients could corrupt the shared representation.

**Solution: stop_gradient_heads**

```python
if stop_gradient_heads and name in stop_gradient_heads:
    results[name] = head(shared.detach())  # Gradients STOP here
else:
    results[name] = head(shared)            # Gradients flow back
```

`shared.detach()` creates a tensor that shares storage but has `requires_grad=False`. The head's own weights still receive gradients from its loss, but no gradient flows back through the shared layer into fusion and encoders.

**Currently:** `STOP_GRADIENT_HEADS = []` (empty — no heads are detached)

---

### 9.4 The 9 Attributes

| Attribute | Classes | Labels | Loss Weight |
|-----------|---------|--------|-------------|
| theme | 9 | Automotive, Education, Fashion, Finance, Food, Gaming, Home, Tech, Travel | 1.0 |
| sentiment | 3 | Negative, Neutral, Positive | 1.5 |
| emotion | 5 | Anger, Excitement, Fear, Joy, Trust | 1.5 |
| dominant_colour | 10 | Red, Black, Blue, Green, White, Grey, Yellow, Brown, Orange, Purple | 1.0 |
| attention_score | 3 | High, Low, Medium | 0.05 |
| trust_safety | 3 | Questionable, Safe, Unsafe | 1.5 |
| predicted_ctr | 3 | High, Low, Medium | 0.05 |
| likelihood_shares | 3 | High, Low, Medium | 0.05 |

**Why loss weights vary:**
- **1.5 (up-weighted):** sentiment, emotion, trust_safety — important for advertisers, clearer signal
- **1.0 (standard):** theme, dominant_colour — clear visual/textual signal
- **0.05 (down-weighted):** attention_score, predicted_ctr, likelihood_shares — essentially random noise (Cramer's V ~ 0.02)

---

## 10. Training Pipeline — Complete Mechanics

### 10.1 Data Loading

**Dataset:** Custom PyTorch Dataset (`lib/preprocessing/dataset.py`)

```python
class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir, label_maps, image_pipeline, text_pipeline):
        self.data = pd.read_csv(csv_path)
        self.label_columns = [col for col in label_maps if col in self.data.columns]
        # Precompute label-to-index maps for O(1) lookup
        self._label_indices = {col: {label: idx for idx, label in enumerate(labels)}
                               for col, labels in label_maps.items()}
```

**Each `__getitem__` returns:**
```
(image_tensor, label_dict)                          — without text
(image_tensor, label_dict, input_ids, attention_mask) — with text
```

---

### 10.2 Image Preprocessing

**Pipeline order:** resize → augment → normalize → tensor

```python
def build_image_transform(size=(512, 512), augment=False, aug_cfg=None):
    def transform(image: np.ndarray) -> torch.Tensor:
        image = resize_image(image, size)           # (H, W, C) → (512, 512, C)
        if augment:
            image = augment_image(image, aug_cfg)   # (512, 512, C) augmented
        image = normalize_image(image)              # (512, 512, C) float32
        return torch.from_numpy(image.transpose(2, 0, 1)).float()  # (C, 512, 512)
```

**ImageNet normalization:**
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB channel stds

# Step 1: Scale to [0, 1]
image = image.astype(np.float32) / 255.0

# Step 2: Normalize
image = (image - IMAGENET_MEAN) / IMAGENET_STD
```

**Why ImageNet normalization?**
The pretrained ResNet-18 was trained on ImageNet with these exact mean/std values. Using different normalization would shift the distribution and degrade pretrained feature quality.

---

### 10.3 Text Preprocessing

**Pipeline:** clean → tokenize

```python
# 1. Clean (light ad-copy cleaner)
cleaned = clean_adcopy(raw_text)
# Steps: Unicode NFC → lowercase → remove control chars → collapse whitespace → strip

# 2. Tokenize
encoding = tokenize_text(cleaned, max_length=256)
# Returns: {"input_ids": (256,), "attention_mask": (256,)}
```

---

### 10.4 Label Encoding

Labels are strings in CSV (e.g., "Food", "Positive"). They must be converted to integers:

```python
# Precomputed at dataset creation:
label_indices = {"theme": {"Automotive": 0, "Education": 1, ..., "Food": 4, ...}}

# At __getitem__ time:
label = row["theme"]           # "Food"
label_idx = label_indices["theme"]["label"]  # 4
label_tensor = torch.tensor(4, dtype=torch.long)
```

---

### 10.5 DataLoader Configuration

```python
# Training loader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,          # Random order each epoch
    drop_last=True,        # Drop incomplete batch (prevents BatchNorm crash)
    num_workers=8,         # Parallel data loading
    pin_memory=True        # Faster GPU transfer
)

# Validation loader
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,         # Consistent order for reproducibility
    drop_last=False,       # Evaluate on all samples
    num_workers=8,
    pin_memory=True
)
```

**Why `drop_last=True` for training?**
If the last batch has only 1 sample, BatchNorm (used in ResNet) computes statistics over a single sample, which is meaningless and can cause numerical instability.

---

### 10.6 Augmentation

Applied only during training, on raw [0, 255] pixels:

```python
# Augmentations applied in order, each independently random:

1. Horizontal flip (50% probability)
   → Image.FLIP_LEFT_RIGHT

2. Vertical flip (10% probability)
   → Image.FLIP_TOP_BOTTOM

3. Random rotation (40% probability)
   → angle = uniform(-15, +15) degrees
   → Image.rotate(angle, resample=BILINEAR)

4. Color jitter (each independently 50% prob, hue 30%):
   → Brightness: factor uniform(0.7, 1.3) → ImageEnhance.Brightness
   → Contrast: factor uniform(0.7, 1.3) → ImageEnhance.Contrast
   → Saturation: factor uniform(0.8, 1.2) → ImageEnhance.Color
   → Hue: factor uniform(-0.05, +0.05) → True HSV rotation

5. Random resized crop (50% probability)
   → scale = uniform(0.8, 1.0)
   → ratio = uniform(0.85, 1.15)
   → crop to scale*ratio, resize back to original
```

**Why these specific augmentations?**
- **Horizontal flip:** Ads can be mirrored without changing meaning
- **No vertical flip:** Ads are almost never upside-down
- **Small rotation (15°):** Ads are usually upright; too much rotation distorts text
- **Color jitter:** Simulates different monitors, printing variations
- **Random crop:** Simulates different framing/cropping of the ad

**Why augment on [0, 255] pixels instead of normalized?**
- Normalized values can go negative, causing artifacts with some augmentations
- Brightness/contrast adjustments work intuitively on [0, 255] range
- Normalization is applied after augmentation

---

### 10.7 Class Weights

If training data is imbalanced (e.g., 70% Positive, 20% Neutral, 10% Negative), the model would bias toward the majority class.

**Solution:** Inverse frequency weighting

```python
weight[class] = total_samples / (num_classes × count[class])
```

**Example:**
```
Sentiment distribution:
  Positive: 3000 samples
  Neutral: 1200 samples
  Negative: 660 samples
  Total: 4860

weight[Positive] = 4860 / (3 × 3000) = 0.54
weight[Neutral]  = 4860 / (3 × 1200) = 1.35
weight[Negative] = 4860 / (3 × 660)  = 2.45
```

The model penalizes misclassifying rare classes more heavily, forcing it to learn them equally.

---

### 10.8 Optimizer — AdamW

**Adam** (Adaptive Moment Estimation) maintains two moving averages per parameter:
- `m`: first moment (gradient mean) — like momentum
- `v`: second moment (gradient squared mean) — like adaptive learning rate

**AdamW** adds proper weight decay:
```
weight_decay: w = w - lr * (gradient / (sqrt(v) + eps) + weight_decay * w)
```

**Key parameters:**
```python
optimizer = AdamW(param_groups, lr=1e-4, weight_decay=0.02)
```

---

### 10.9 Parameter Grouping

Different parameters need different learning rates:

```python
param_groups = [
    # Group 1: Encoder parameters (pretrained, need small LR)
    {"params": encoder_decay_params, "lr": 1.5e-5, "weight_decay": 0.02},
    {"params": encoder_no_decay_params, "lr": 1.5e-5, "weight_decay": 0.0},
    
    # Group 2: Other parameters (random init, need larger LR)
    {"params": other_decay_params, "lr": 1e-4, "weight_decay": 0.02},
    {"params": other_no_decay_params, "lr": 1e-4, "weight_decay": 0.0},
]
```

**No-decay parameters:** bias, LayerNorm.weight, LayerNorm.bias
- These should NOT have weight decay (it would hurt training)
- Weight decay penalizes large weights; biases and norms need flexibility

**Why different learning rates?**
- Pretrained encoder already knows how to process text → small changes only (LR = 1.5e-5)
- New layers (fusion, heads) start random → need larger updates (LR = 1e-4)

---

### 10.10 Learning Rate Scheduler

**Phase 1: Warmup (5 epochs)**
```python
LinearLR(start_factor=1e-5, end_factor=1.0, total_iters=5)
```
- LR starts at `1e-5 × base_lr` (nearly zero)
- Linearly increases to `base_lr` over 5 epochs
- **Why:** Prevents large initial gradients from destabilizing training

**Phase 2: Cosine Annealing (95 epochs)**
```python
CosineAnnealingLR(T_max=95, eta_min=1e-6)
```
- LR follows cosine curve: `eta_min + 0.5 × (base_lr - eta_min) × (1 + cos(π × t / T_max))`
- Starts at `base_lr`, ends at `eta_min = 1e-6`
- **Why:** Smooth decay that avoids abrupt LR drops

**Combined via SequentialLR:**
```python
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
```

---

### 10.11 Loss Function

**Cross-Entropy Loss** for each attribute:

```python
loss = -sum_i(y_i * log(softmax(logits)_i))
```

Where:
- `y` = one-hot encoded true label
- `logits` = raw model output (before softmax)
- `softmax(logits)` = probability distribution

**Multi-attribute total loss:**
```python
total_loss = sum(weight[attr] * CEloss(outputs[attr], labels[attr]) for attr in attributes)
```

---

### 10.12 Label Smoothing

```python
CrossEntropyLoss(label_smoothing=0.25)
```

**What it does:**
Instead of hard targets `[0, 0, 1, 0]` (100% confident in class 2), uses soft targets `[0.0625, 0.0625, 0.8125, 0.0625]` (81.25% confident, 6.25% spread to other classes).

**Formula:**
```
smoothed_target = (1 - epsilon) * one_hot + epsilon / num_classes
```

**Why?**
- Prevents the model from becoming overconfident
- Acts as regularization (similar to dropout)
- Improves generalization
- 0.25 is relatively high — aggressive smoothing for small dataset

---

### 10.13 Mixup Augmentation

**What it does:**
Blends two training examples together:

```python
lam = Beta(alpha, alpha).sample()  # Random mixing ratio
mixed_images = lam * images_a + (1 - lam) * images_b
mixed_labels = lam * labels_a + (1 - lam) * labels_b
```

**Loss computation:**
```python
loss = lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
```

**Why Mixup?**
- Creates virtual training examples that lie between real ones
- Forces the model to behave linearly between examples
- Reduces overfitting and adversarial vulnerability
- **Currently disabled** (`mixup_alpha = 0.0`) because multi-task Mixup is complex

---

### 10.14 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What it does:**
If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally.

**Why?**
- Prevents exploding gradients (gradients that become extremely large)
- Stabilizes training, especially with Transformers
- max_norm=1.0 is standard for Transformer training

---

### 10.15 Training Loop — Step by Step

```
For epoch in range(100):

    === TRAIN PHASE ===
    model.train()  # Enable dropout, batch norm training mode
    
    for batch in train_loader:
        # 1. Unpack
        images = batch[0].to(device)          # (B, 3, 512, 512)
        labels = batch[1]                      # dict of (B,) tensors
        input_ids = batch[2].to(device)        # (B, 256)
        attention_mask = batch[3].to(device)   # (B, 256)
        
        # 2. Mixup (if enabled)
        if mixup_alpha > 0:
            lam = Beta(mixup_alpha, mixup_alpha).sample()
            perm = randperm(B)
            images = lam * images + (1-lam) * images[perm]
            labels_a, labels_b = labels, {k: v[perm] for k,v in labels.items()}
        
        # 3. Forward pass
        optimizer.zero_grad()                  # Clear old gradients
        outputs = model(images, input_ids, attention_mask)  # Dict of (B, C) logits
        
        # 4. Compute loss
        total_loss = 0
        for attr in attributes:
            weight = attribute_loss_weights[attr]
            loss = mixup_criterion(criterion, outputs[attr], labels_a[attr], labels_b[attr], lam)
            total_loss += weight * loss
        
        # 5. Backward pass
        total_loss.backward()                  # Compute gradients
        
        # 6. Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 7. Optimizer step
        optimizer.step()                       # Update weights
        
        # 8. Accumulate metrics
        running_loss += total_loss.item() * B
        correct += (outputs[attr].argmax(1) == labels_a[attr]).sum().item()
    
    avg_train_loss = running_loss / total_samples
    train_accuracy = correct / total_samples
    
    === VALIDATION PHASE ===
    model.eval()  # Disable dropout, batch norm uses running stats
    with torch.no_grad():  # Don't compute gradients
        for batch in val_loader:
            # Same forward pass, NO Mixup
            outputs = model(images, input_ids, attention_mask)
            loss = sum(criterion[attr](outputs[attr], labels[attr]) for attr)
            # Accumulate val metrics
    
    avg_val_loss = val_running_loss / val_total_samples
    val_accuracy = val_correct / val_total_samples
    
    === SCHEDULER STEP ===
    scheduler.step()  # Update learning rate
    
    === LOGGING ===
    Log: epoch, train_loss, train_acc, val_loss, val_acc, lr
```

---

### 10.16 Validation Loop

**Same as training but:**
- `model.eval()` — disables dropout, uses batch norm running statistics
- `torch.no_grad()` — disables gradient computation (saves memory and compute)
- No Mixup
- No gradient clipping
- No optimizer step

---

### 10.17 Early Stopping

If validation loss doesn't improve for `early_stopping_patience=10` epochs, training stops.

```
best_val_loss = infinity
patience_counter = 0

for epoch in range(100):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)  # Save best model
    else:
        patience_counter += 1
    
    if patience_counter >= 10:
        print("Early stopping!")
        break
```

**Why early stopping?**
- Prevents overfitting (model starts memorizing training data)
- Saves compute (no point training if validation isn't improving)
- The saved best model is the one with lowest validation loss

---

## 11. Inference Pipeline — End to End

### 11.1 Prediction Pipeline Construction

```python
def build_prediction_pipeline(config_path, checkpoint_path):
    # 1. Load config
    config = load_config(config_path)
    
    # 2. Build label maps
    label_maps = get_label_maps(config)
    
    # 3. Create and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg, device, checkpoint_path)
    # Creates FG_MFN, loads weights, sets eval mode
    
    # 4. Create OCR engine
    ocr_engine = create_ocr_engine("easyocr", Path("local/ocr"), languages=["en"])
    
    # 5. Build predictor
    predictor = Predictor(model, label_maps)
    
    return predictor
```

---

### 11.2 Single Image Prediction

```python
def predict_image(image, model, ocr_engine, label_maps, filename, config, predictor):
    # 1. OCR extraction
    raw_text, confidence = extract_text(image, ocr_engine)
    # raw_text: "buy now 50 off premium headphones"
    
    # 2. Text cleaning
    cleaned_text = clean_text(raw_text)
    # cleaned_text: "buy now 50 off premium headphones"
    
    # 3. Image preparation
    image_tensor = _prepare_image_tensor(image, (512, 512))
    # Shape: (1, 3, 512, 512) float32, ImageNet-normalized
    
    # 4. Text tokenization
    input_ids, attention_mask = _prepare_text_tensors(cleaned_text, 256)
    # Shapes: (1, 256) long each
    
    # 5. Model inference
    result = predictor.predict_single(image_tensor, input_ids, attention_mask)
    # Returns: {"theme": "Food", "theme_confidence": 0.95, ...}
    
    # 6. Post-processing
    result = format_prediction_result(result)
    # Strips confidence scores, numeric indices
    
    # 7. Enrich with metadata
    result["ocr_text"] = raw_text
    result["filename"] = filename
    result["keywords"] = extract_keywords(raw_text)
    result["monetary_mention"] = extract_monetary_mention(raw_text)
    result["call_to_action"] = extract_call_to_action(raw_text)
    result["object_detected"] = extract_objects_mentioned(raw_text)
    
    return result
```

---

### 11.3 Batch Prediction

```python
def predict_batch(images, model, ocr_engine, label_maps, filenames, config, predictor):
    results = []
    predictor = predictor or Predictor(model, label_maps)  # Create once
    
    for idx, image in enumerate(images):
        try:
            result = predict_image(
                image, model, ocr_engine, label_maps,
                filename=filenames[idx] if filenames else "",
                config=config,
                predictor=predictor  # Reuse predictor
            )
            results.append(result)
        except MemoryError:
            raise  # OOM is fatal
        except Exception as e:
            results.append({"error": str(e), "filename": filenames[idx]})
    
    return results
```

**Note:** This processes images sequentially, not in parallel batches. Each image goes through OCR → tokenize → model independently.

---

### 11.4 Post-Processing

**Function:** `format_prediction_result(raw_result)` (`lib/services/postprocessor.py`)

**Input:**
```python
{
    "theme": "Food",
    "theme_confidence": 0.95,
    "theme_predicted_label_num": 4,
    "sentiment": "Positive",
    "sentiment_confidence": 0.87,
    "sentiment_predicted_label_num": 2,
    ...
}
```

**Output:**
```python
{
    "theme": "Food",
    "sentiment": "Positive",
    ...
}
```

**What gets stripped:**
- Keys ending with `_confidence`, `_score`, `_accuracy`, `_f1` (unless `include_confidence=True`)
- Keys ending with `_predicted_label_num`
- Exact matches: `confidence_score`, `predicted_label_num`

---

## 12. Web Application

### 12.1 FastAPI Server

```python
app = FastAPI(title="Multi-Model Prediction API", version="2.0.0", lifespan=lifespan)
```

**Startup (lifespan):**
1. Load config from `configs/model/model_config.json`
2. Find checkpoint (env var → config → `saved_models/best_model_*.pt` → `last_model.pt`)
3. Build prediction pipeline (model + OCR + predictor)
4. Configure predictor and OCR engine
5. Enable CORS middleware

**Shutdown:**
1. Clean up upload directory

---

### 12.2 API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/predict` | Upload images, get predictions |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| GET | `/` | Service info, all routes |
| POST | `/auth/register` | User registration |
| POST | `/auth/login` | User login |
| POST | `/auth/refresh` | Token refresh |
| POST | `/auth/logout` | User logout |
| GET | `/history` | Prediction history |
| GET | `/analytics` | Attribute distributions |
| POST | `/api-keys` | Create API key |
| DELETE | `/api-keys/{id}` | Delete API key |
| POST | `/admin/tenants` | Create tenant |
| POST | `/invites` | Send team invite |

---

### 12.3 Authentication (Supabase)

**Feature-flagged:** Only active when `AUTH_ENABLED=true`

**When auth is enabled:**
- Every request requires a valid JWT token
- `require_auth` dependency extracts `tenant_id` and `user_id`
- Quota checking: each tenant has a monthly prediction limit
- Predictions are saved to Supabase with tenant/user association

**When auth is disabled:**
- Default tenant/user IDs are used
- No quota checking
- Predictions are not saved

---

### 12.4 Request Flow

```
Client sends POST /predict with image file(s)
    |
    v
predict_endpoint() [app/predict.py]
    |
    v
require_auth dependency → RequestContext (tenant_id, user_id)
    |
    v
_check_quota(tenant_id) if auth enabled
    → queries Supabase for usage
    → if remaining <= 0 → HTTPException(429)
    |
    v
For each uploaded file:
    |
    v
_load_image_from_upload(upload_file) → PIL.Image (RGB)
    |
    v
predict_image(image, model, ocr_engine, label_maps, filename, predictor)
    |
    v
extract_text(image, ocr_engine) → (raw_text, confidence)
    |
    v
clean_text(raw_text) → cleaned_text
    |
    v
_prepare_image_tensor(image, (512, 512)) → (1, 3, 512, 512)
    |
    v
_prepare_text_tensors(cleaned_text, 256) → (1, 256), (1, 256)
    |
    v
predictor.predict_single(image_tensor, input_ids, attention_mask)
    → model.forward() → raw_outputs
    → softmax → argmax per attribute
    → label lookup from label_maps
    → {attr: label, attr_confidence: float, attr_predicted_label_num: int}
    |
    v
format_prediction_result() → strip scores and numeric indices
    |
    v
Enrich: ocr_text, filename, keywords, monetary_mention, call_to_action, object_detected
    |
    v
_save_prediction() if auth enabled (non-blocking)
    |
    v
Return {predictions: [...], total_images: N, processing_time_ms: T}
```

---

### 12.5 Error Handling

| Error | HTTP Code | Condition |
|-------|-----------|-----------|
| Service not configured | 503 | Model/OCR not loaded |
| Quota exceeded | 429 | Monthly limit reached |
| Invalid image | 400 | IOError loading file |
| Prediction failure | 400 | RuntimeError in model |
| Quota check failure | 500 | Supabase query failed (fail-closed) |
| Oversized upload | 413 | File > 10 MB |
| Bad extension | 400 | Not in {png, jpg, jpeg, gif, bmp, tiff, webp} |
| MemoryError | 500 | OOM during batch (fatal, re-raised) |
| Other exceptions | 200 | Caught per-image, returned as `{"error": msg}` |

---

## 13. Complete End-to-End Dimension Trace

```
BATCH SIZE: 16

INPUT:
  images:        (16, 3, 512, 512)     — 16 RGB images, 512x512 pixels
  input_ids:     (16, 256)              — tokenized text, 256 tokens each
  attention_mask: (16, 256)             — 1=real token, 0=padding

STEP 1: VisualModule
  ResNet-18 (frozen, stripped):
    conv1:       (16, 3, 512, 512) → (16, 64, 256, 256)
    bn1+relu:    (16, 64, 256, 256)
    maxpool:     (16, 64, 128, 128)
    layer1:      (16, 64, 128, 128)  → (16, 64, 128, 128)
    layer2:      (16, 64, 128, 128)  → (16, 128, 64, 64)
    layer3:      (16, 128, 64, 64)   → (16, 256, 32, 32)
    layer4:      (16, 256, 32, 32)   → (16, 512, 16, 16)
    avgpool:     (16, 512, 1, 1)
    Identity():  (16, 512, 1, 1)
  flatten:       (16, 512)
  OUTPUT: visual_features (16, 512)

STEP 2: TextModule
  DistilBERT (eager attention):
    Embeddings:  (16, 256) → (16, 256, 768)
    Transformer ×6:
      Layer 0:   (16, 256, 768) → (16, 256, 768)
      Layer 1:   (16, 256, 768) → (16, 256, 768)
      Layer 2:   (16, 256, 768) → (16, 256, 768)
      Layer 3:   (16, 256, 768) → (16, 256, 768)
      Layer 4:   (16, 256, 768) → (16, 256, 768)
      Layer 5:   (16, 256, 768) → (16, 256, 768)
    Attentions:  tuple of 6 × (16, 12, 256, 256)
    last_hidden: (16, 256, 768)
  
  Attention-weighted pooling:
    attentions[-1]:  (16, 12, 256, 256)
    mean(dim=1):     (16, 256, 256)     — average across 12 heads
    sum(dim=1):      (16, 256)           — per-token importance
    * mask:          (16, 256)           — zero padding
    normalize:       (16, 256)           — sum=1 per sample
    weighted sum:    (16, 768)           — final pooled vector
  OUTPUT: text_features (16, 768)

STEP 3: CrossModalAttention
  visual_proj: Linear(512, 768)
    (16, 512) → (16, 768) → unsqueeze → (16, 1, 768)
  
  text_proj: Linear(768, 768)
    (16, 768) → (16, 768) → unsqueeze → (16, 1, 768)
  
  v2t_attn (visual attends to text):
    Q=(16,1,768), K=(16,1,768), V=(16,1,768)
    8 heads × 96-d each
    attention: (16, 8, 1, 1) → softmax → (16, 8, 1, 1)
    output: (16, 8, 1, 96) → concat → (16, 1, 768) → proj → (16, 1, 768)
    dropout(0.5): (16, 1, 768)
  
  t2v_attn (text attends to visual):
    Same: (16, 1, 768)
  
  Residual + LayerNorm:
    v + v_attended: (16, 1, 768)
    v_norm: (16, 1, 768) → squeeze → (16, 768)
    t + t_attended: (16, 1, 768)
    t_norm: (16, 1, 768) → squeeze → (16, 768)
  
  Concatenate:
    cat: (16, 1536)
  
  fusion_proj: Linear(1536, 512)
    (16, 1536) → (16, 512)
  OUTPUT: fused (16, 512)

STEP 4: Shared FC (deep)
  Linear(512, 512):  (16, 512) → (16, 512)
  GELU:              (16, 512) → (16, 512)
  Dropout(0.5):      (16, 512) → (16, 512)
  Linear(512, 512):  (16, 512) → (16, 512)
  GELU:              (16, 512) → (16, 512)
  Dropout(0.25):     (16, 512) → (16, 512)
  OUTPUT: shared (16, 512)

STEP 5: Classification Heads
  Each: Linear(512, 256) → GELU → Dropout(0.5) → Linear(256, C)
  
  theme:             (16, 512) → (16, 256) → (16, 9)
  sentiment:         (16, 512) → (16, 256) → (16, 3)
  emotion:           (16, 512) → (16, 256) → (16, 5)
  dominant_colour:   (16, 512) → (16, 256) → (16, 10)
  attention_score:   (16, 512) → (16, 256) → (16, 3)
  trust_safety:      (16, 512) → (16, 256) → (16, 3)
  predicted_ctr:     (16, 512) → (16, 256) → (16, 3)
  likelihood_shares: (16, 512) → (16, 256) → (16, 3)

FINAL OUTPUT: Dict[str, Tensor] with 8 entries, total 39 logit values per sample
```

---

## 14. Hyperparameter Reference

### Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| IMAGE_BACKBONE | `resnet18` | config |
| TEXT_ENCODER | `distilbert-base-uncased` | config |
| TEXT_POOLING | `attention_weighted` | config |
| FUSION_TYPE | `attention` | config |
| HIDDEN_DIM | 512 | config |
| ATTENTION_DIM | 768 | config |
| ATTENTION_HEADS | 8 | config |
| DROPOUT | 0.5 | config |
| FREEZE_BACKBONE | true | config |
| DEEP_SHARED_LAYER | true | config |

### Training

| Parameter | Value | Source |
|-----------|-------|--------|
| learning_rate | 1e-4 | config |
| encoder_learning_rate | 1.5e-5 | config |
| weight_decay | 0.02 | config |
| batch_size | 16 | config |
| epochs | 100 | config |
| warmup_epochs | 5 | config |
| label_smoothing | 0.25 | config |
| early_stopping_patience | 10 | config |
| scheduler_type | `cosine` | config |
| scheduler_eta_min | 1e-6 | config |
| mixup_alpha | 0.0 (disabled) | config |
| gradient_clip_max_norm | 1.0 | hardcoded |

### Image

| Parameter | Value | Source |
|-----------|-------|--------|
| image_size | (512, 512) | config |
| IMAGENET_MEAN | [0.485, 0.456, 0.406] | hardcoded |
| IMAGENET_STD | [0.229, 0.224, 0.225] | hardcoded |
| horizontal_flip_prob | 0.5 | config |
| vertical_flip_prob | 0.0 | config |
| rotation_degrees | 10 | config |
| brightness | 0.3 | config |
| contrast | 0.3 | config |
| saturation | 0.2 | config |
| hue | 0.05 | config |

### Text

| Parameter | Value | Source |
|-----------|-------|--------|
| text_max_length | 256 | config |
| tokenizer | distilbert-base-uncased | config |
| MAX_RAW_TEXT_LENGTH | 100,000 | hardcoded |

### Loss Weights

| Attribute | Weight |
|-----------|--------|
| theme | 1.0 |
| sentiment | 1.5 |
| emotion | 1.5 |
| dominant_colour | 1.0 |
| attention_score | 0.05 |
| trust_safety | 1.5 |
| predicted_ctr | 0.05 |
| likelihood_shares | 0.05 |

---

## 15. Design Decisions — Why Each Choice

### Why pretrained models instead of training from scratch?

**Data scarcity:** ~4,860 images is far too few to train a CNN from scratch (needs millions). Pretrained models (trained on ImageNet's 14M images and Wikipedia's billions of words) provide a strong starting point.

### Why freeze the ResNet backbone?

**Overfitting prevention:** 11.2M unfreezing parameters with only 4,860 images → memorization. Freezing preserves pretrained knowledge while reducing trainable parameters to ~74.4M.

### Why Cross-Modal Attention instead of concatenation?

**Explicit alignment:** Attention learns which visual-text pairs are meaningful ("red" ↔ "sale"), while concatenation leaves this discovery to deeper layers (requires more data and capacity).

### Why 8 attention heads?

**Empirical balance:** Original Transformer paper (Vaswani et al., 2017) used 8. Too few misses diverse relationships; too many dilutes each head's information.

### Why DistilBERT instead of full BERT?

**Efficiency:** 40% smaller, 60% faster, 97% as accurate. For a real-time prediction API, speed matters.

### Why separate classification heads?

**Specialization:** Each attribute has different class counts (3-10) and different semantics. Separate heads allow independent capacity and prevent interference.

### Why per-attribute loss weighting?

**Signal quality:** Engagement metrics (CTR, shares) are essentially noise (Cramer's V ~ 0.02). Without down-weighting, the model would waste capacity trying to predict noise.

### Why AdamW?

**Transformer standard:** Adam adapts learning rates per-parameter; AdamW adds proper weight decay. It's the default for Transformer training.

### Why cosine annealing?

**Smooth decay:** Avoids abrupt LR drops that can cause training instability. The cosine curve provides a natural decay schedule.

### Why label smoothing (0.25)?

**Prevents overconfidence:** With a small dataset, the model can memorize training labels with high confidence. Smoothing forces uncertainty, improving generalization.

### Why gradient clipping (max_norm=1.0)?

**Stability:** Transformers are prone to gradient explosions. Clipping prevents catastrophic updates.

### Why different dropout rates in shared FC?

**Information flow:** First layer sees raw fused features (noisier → higher dropout 0.5); second layer sees refined features (cleaner → lower dropout 0.25).

### Why attention-weighted pooling?

**Leverages model knowledge:** The Transformer already computes attention scores. Using them as pooling weights gives a more meaningful representation than mean/CLS pooling.

---

## 16. Terminology Glossary

| Term | Definition |
|------|------------|
| **Activation function** | Non-linear function (GELU, ReLU) applied after linear transformation to introduce non-linearity |
| **AdamW** | Optimizer that adapts learning rates per-parameter with proper weight decay |
| **Attention** | Mechanism computing relevance scores between elements (Q, K, V) |
| **Backbone** | The base feature extractor (ResNet-18) before any task-specific layers |
| **Backpropagation** | Algorithm computing gradients by chaining derivatives backward through the network |
| **Batch** | Group of samples processed together (batch_size=16) |
| **BatchNorm** | Normalizes activations within each batch for training stability |
| **CNN** | Convolutional Neural Network — grid-data specialist using learnable filters |
| **CLS token** | Special token prepended to sequences; its output used as sentence representation |
| **Cross-entropy** | Loss function measuring distance between predicted and true probability distributions |
| **Dropout** | Randomly zeroing activations during training to prevent co-adaptation |
| **Dropout rate** | Fraction of activations zeroed (0.5 = 50%) |
| **Embedding** | Mapping discrete tokens to continuous vectors |
| **Encoder** | Neural network converting raw data to feature vectors |
| **Epoch** | One complete pass through all training data |
| **Exploding gradients** | Gradients becoming extremely large, causing unstable updates |
| **Feature vector** | List of numbers representing the meaning of input data |
| **FG_MFN** | Fine-Grained Multi-Modal Fusion Network — this project's architecture |
| **Fully connected** | Layer connecting every input neuron to every output neuron |
| **GELU** | Gaussian Error Linear Unit — smooth activation function |
| **Gradient** | Direction and magnitude of weight update needed to reduce loss |
| **Gradient clipping** | Limiting gradient norm to prevent exploding gradients |
| **Label smoothing** | Softening hard one-hot targets to prevent overconfidence |
| **Layer Normalization** | Normalizing activations across features (not across batch) |
| **Learning rate** | Step size for weight updates (1e-4 default) |
| **Loss** | Measure of how wrong predictions are (lower = better) |
| **LR scheduler** | Algorithm adjusting learning rate during training |
| **MLP** | Multi-Layer Perceptron — stack of fully connected layers |
| **Multi-head attention** | Parallel attention functions with different learned projections |
| **Multi-modal** | Processing multiple data types (image + text) |
| **OCR** | Optical Character Recognition — extracting text from images |
| **Overfitting** | Model memorizing training data instead of learning general patterns |
| **Pooling** | Collapsing sequence outputs to a single vector |
| **Pretrained** | Already trained on large datasets (ImageNet, Wikipedia) |
| **Projection** | Linear transformation changing dimensionality |
| **Residual connection** | Skip connection adding input directly to output |
| **ResNet** | Residual Network — CNN with skip connections |
| **Self-attention** | Attention where Q, K, V all come from the same source |
| **Softmax** | Converts logits to probability distribution (sums to 1) |
| **Stop-gradient** | Detaching tensor to prevent gradient flow |
| **Tokenization** | Splitting text into subword units and converting to integer IDs |
| **Transformer** | Architecture using self-attention for sequence processing |
| **Vanishing gradients** | Gradients becoming extremely small, preventing learning |
| **Weight decay** | L2 regularization penalizing large weights |
| **Weights** | Learnable parameters inside neural network layers |
