# Full Model Theory: How FG_MFN Works — From Image to Output

> **Purpose:** Understand every single detail of how the Fine-Grained Multi-Modal Fusion Network (FG_MFN) processes an image and text, fuses them, and produces 9 attribute predictions. Every equation, every layer, every dimension.

---

## 📚 Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Input: What Goes Into the Model](#2-input-what-goes-into-the-model)
3. [Stage 1: Visual Feature Extraction (ResNet-50)](#3-stage-1-visual-feature-extraction-resnet-50)
4. [Stage 2: Text Feature Extraction (DistilBERT)](#4-stage-2-text-feature-extraction-distilbert)
5. [Stage 3: Cross-Modal Fusion (Multi-Head Attention)](#5-stage-3-cross-modal-fusion-multi-head-attention)
6. [Stage 4: Feature Concatenation & Projection](#6-stage-4-feature-concatenation--projection)
7. [Stage 5: Multi-Task Prediction Heads](#7-stage-5-multi-task-prediction-heads)
8. [Stage 6: Loss Computation](#8-stage-6-loss-computation)
9. [Stage 7: Backpropagation & Optimization](#9-stage-7-backpropagation--optimization)
10. [Stage 8: Inference (Prediction)](#10-stage-8-inference-prediction)
11. [Complete Mathematical Walkthrough](#11-complete-mathematical-walkthrough)
12. [Why This Architecture Works](#12-why-this-architecture-works)

---

## 1. The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FG_MFN ARCHITECTURE                           │
│                                                                      │
│   ┌──────────────┐                                                   │
│   │   Image      │──→ ResNet-50 ──→ Visual Features (2048-d)         │
│   │  (224×224×3) │                    ↓                             │
│   └──────────────┘              Project to 768-d                     │
│                                  ↓                                   │
│                              ┌───────────┐                           │
│                              │  Fusion   │                           │
│   ┌──────────────┐           │  Module   │                           │
│   │   Text       │──→ DistilBERT ──→ Text Features (768-d)           │
│   │  (256 tokens)│              ↓                                   │
│   └──────────────┘         ┌───────────┐                             │
│                            │ 9 Heads   │                             │
│                            │ (themes,  │                             │
│                            │ sentiment,│                             │
│                            │ emotion,  │                             │
│                            │ ...)      │                             │
│                            └───────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

**The model has 4 main components:**
1. **Visual Encoder** (ResNet-50): Extracts visual features from the image
2. **Text Encoder** (DistilBERT): Extracts textual features from the ad copy
3. **Fusion Module** (Cross-Modal Attention): Combines visual and text features
4. **Prediction Heads** (9 MLPs): Predicts 9 different attributes

---

## 2. Input: What Goes Into the Model

The model receives **two inputs**:

### Image Input
- **Shape:** `(batch_size, 3, 224, 224)`
- **Type:** Float tensor
- **Values:** Normalized to `[-2.12, 2.64]` (after ImageNet normalization)
- **Meaning:** RGB image with 3 channels (Red, Green, Blue), 224 pixels wide, 224 pixels tall

### Text Input
- **Shape:** `(batch_size, 256)` — input IDs
- **Shape:** `(batch_size, 256)` — attention mask
- **Type:** Long tensor (integers)
- **Values:** Token IDs from DistilBERT's vocabulary (0-30,000)
- **Meaning:** Each ad text converted to 256 tokens

**Example:**
```
Ad text: "Buy 2 Get 1 Free! Limited time offer."
Token IDs: [101, 2378, 1016, 2053, 1015, 2489, 999, 102, 0, 0, ...]
            CLS  buy   2    get   1    free  !   SEP PAD PAD ...
```

---

## 3. Stage 1: Visual Feature Extraction (ResNet-50)

### What is ResNet-50?

**ResNet-50** is a 50-layer Convolutional Neural Network (CNN) pretrained on ImageNet (1.2 million images, 1000 classes).

**Key innovation:** **Residual connections** (skip connections) that allow training very deep networks without vanishing gradients.

### The Residual Block

```
┌─────────────────────────────────────────┐
│           Residual Block                │
│                                          │
│   Input (x)                              │
│      ↓                                   │
│   Conv 1×1 (reduce dimensions)           │
│      ↓                                   │
│   BatchNorm + ReLU                       │
│      ↓                                   │
│   Conv 3×3 (extract features)            │
│      ↓                                   │
│   BatchNorm + ReLU                       │
│      ↓                                   │
│   Conv 1×1 (restore dimensions)          │
│      ↓                                   │
│   BatchNorm                              │
│      ↓                                   │
│   Add: x + F(x)  ← THE KEY IDEA         │
│      ↓                                   │
│   ReLU                                   │
│      ↓                                   │
│   Output                                 │
└─────────────────────────────────────────┘
```

**The equation:**
$$y = F(x, \{W_i\}) + x$$

Where:
- $x$ = input
- $F(x, \{W_i\})$ = residual mapping (what the conv layers learn)
- $y$ = output

**Why this works:**
Instead of learning $H(x) = F(x) + x$, the layers learn the **residual** $F(x) = H(x) - x$. This is easier to optimize because:
- If the optimal mapping is identity, $F(x) = 0$ is easy to learn
- Gradients flow directly through the skip connection (no vanishing)

### ResNet-50 Architecture (Simplified)

```
Input Image (224×224×3)
    ↓
Conv 7×7, 64 filters, stride 2
    ↓ (112×112×64)
MaxPool 3×3, stride 2
    ↓ (56×56×64)
Residual Block × 3 (64 filters)
    ↓ (56×56×64)
Residual Block × 4 (128 filters)
    ↓ (28×28×128)
Residual Block × 6 (256 filters)
    ↓ (14×14×256)
Residual Block × 3 (512 filters)
    ↓ (7×7×2048)
Global Average Pooling
    ↓ (2048-d vector)
```

**Total layers:** 50 (49 conv + 1 FC)

### What Happens in Our Model

```python
# From lib/models/fg_mfn.py
visual_features = resnet50(image)  # Shape: (batch_size, 2048)
```

**Step-by-step:**

1. **Convolution layers** extract hierarchical features:
   - **Early layers:** edges, colors, textures
   - **Middle layers:** shapes, patterns, parts
   - **Deep layers:** objects, scenes, concepts

2. **Global Average Pooling** converts the 7×7×2048 feature map into a 2048-dimensional vector:
$$\text{GAP}(x) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j}$$

3. **Output:** A 2048-dimensional vector where each dimension represents a learned visual concept.

**Example:**
```
Visual features might encode:
- Dimension 42: "brightness"
- Dimension 137: "presence of people"
- Dimension 891: "warm color palette"
- Dimension 2047: "text-heavy composition"
```

### Projection to 768-d

The visual features (2048-d) are projected to 768-d to match the text features:

```python
visual_projection = nn.Linear(2048, 768)
visual_features_768 = visual_projection(visual_features)
```

**The equation:**
$$v_{768} = W_v \cdot v_{2048} + b_v$$

Where:
- $W_v$ = learnable weight matrix (768×2048)
- $b_v$ = learnable bias vector (768-d)
- $v_{2048}$ = input visual features (2048-d)
- $v_{768}$ = output projected features (768-d)

**Why 768?** It matches DistilBERT's hidden size, enabling fusion.

---

## 4. Stage 2: Text Feature Extraction (DistilBERT)

### What is DistilBERT?

**DistilBERT** is a compressed version of BERT (Bidirectional Encoder Representations from Transformers):
- **60% smaller** than BERT
- **60% faster**
- **97% of BERT's performance**

**Key innovation:** **Knowledge distillation** — a small model (student) learns to mimic a large model (teacher).

### What is a Transformer?

A Transformer is a neural network architecture based on **self-attention**, introduced in the paper "Attention Is All You Need" (2017).

**Core idea:** Instead of processing sequences step-by-step (like RNNs), Transformers process **all positions in parallel** and use attention to understand relationships between positions.

### Self-Attention Mechanism

**The equation:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ = Query matrix (what we're looking for)
- $K$ = Key matrix (what we have)
- $V$ = Value matrix (the actual information)
- $d_k$ = dimension of keys (64 in our case)

**Intuition:**
- Each token creates a **query** (what information does it need?)
- Each token creates a **key** (what information does it have?)
- Each token creates a **value** (the actual information)
- Attention scores = how much each token should attend to every other token

**Example:**
```
Sentence: "The cat sat on the mat"

Token "sat" might attend strongly to:
- "cat" (who sat?)
- "mat" (where did it sit?)

Token "cat" might attend strongly to:
- "The" (which cat?)
- "sat" (what did the cat do?)
```

### Multi-Head Attention

Instead of one attention mechanism, we use **multiple heads** in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Why multiple heads?**
Each head can learn **different types of relationships**:
- Head 1: syntactic relationships (subject-verb)
- Head 2: semantic relationships (cat-animal)
- Head 3: positional relationships (near, far)
- Head 4: coreference (it → cat)
- ... (8 heads total)

### DistilBERT Architecture

```
Input Tokens (256 tokens)
    ↓
Token Embedding (30522 vocab × 768 dim)
    ↓
Positional Embedding (256 positions × 768 dim)
    ↓
Sum: Token + Position
    ↓
Transformer Block × 6
    ↓ (each block has Multi-Head Attention + Feed-Forward)
    ↓
Output: (batch_size, 256, 768)
```

**Each Transformer Block:**
```
Input (256, 768)
    ↓
Multi-Head Attention (8 heads)
    ↓
Add & Norm (residual connection + layer norm)
    ↓
Feed-Forward Network (768 → 3072 → 768)
    ↓
Add & Norm
    ↓
Output (256, 768)
```

### Attention-Weighted Pooling

Instead of using just the `[CLS]` token, we use **attention-weighted pooling**:

```python
# Attention scores for each token
attention_scores = nn.Linear(768, 1)(text_features)  # (batch, 256, 1)
attention_weights = softmax(attention_scores, dim=1)  # (batch, 256, 1)

# Weighted sum
text_features_pooled = (text_features * attention_weights).sum(dim=1)  # (batch, 768)
```

**The equation:**
$$t_{pooled} = \sum_{i=1}^{256} \alpha_i \cdot t_i$$

Where:
- $t_i$ = token $i$'s features (768-d)
- $\alpha_i$ = attention weight for token $i$
- $\sum \alpha_i = 1$ (softmax ensures this)

**Why attention pooling?**
- Different tokens have different importance.
- "Free!" might be more important than "the".
- Attention learns which tokens matter most.

**Output:** A 768-dimensional vector representing the entire ad text.

---

## 5. Stage 3: Cross-Modal Fusion (Multi-Head Attention)

### The Problem

We have:
- Visual features: $v \in \mathbb{R}^{768}$
- Text features: $t \in \mathbb{R}^{768}$

**Naive approach:** Concatenate them: $[v; t] \in \mathbb{R}^{1536}$

**Problem:** This treats them as independent. But vision and text **interact**:
- The word "Sale" changes how we interpret a red banner
- A product image changes how we interpret "limited"

### Cross-Modal Attention

**Idea:** Let visual features attend to text features, and vice versa.

#### Visual-to-Text Attention

**Query:** Visual features (what does the image want to know?)
**Key & Value:** Text features (what information does the text have?)

$$v' = \text{MultiHeadAttention}(Q=v, K=t, V=t)$$

**Equation:**
$$v' = \text{softmax}\left(\frac{v \cdot t^T}{\sqrt{64}}\right) t$$

**Intuition:** "Given this image, which words are most relevant?"

**Example:**
```
Image: Red banner with "50% OFF"
Text: "Limited time offer, buy now!"

Visual features might attend to:
- "Limited" (the red suggests urgency)
- "buy" (the banner suggests action)
- "now" (the bold text suggests immediacy)
```

#### Text-to-Visual Attention

**Query:** Text features (what does the text want to know?)
**Key & Value:** Visual features (what information does the image have?)

$$t' = \text{MultiHeadAttention}(Q=t, K=v, V=v)$$

**Intuition:** "Given this text, which visual elements are most relevant?"

**Example:**
```
Text: "New summer collection"
Image: Beach scene with models

Text features might attend to:
- Beach background (summer context)
- Models (people wearing clothes)
- Bright colors (summer vibe)
```

#### Combined Fusion

```python
# Bidirectional cross-attention
v_attended, _ = cross_attention(v, t, t)  # Visual attends to text
t_attended, _ = cross_attention(t, v, v)  # Text attends to visual

# Concatenate
fused = torch.cat([v_attended, t_attended], dim=-1)  # (batch, 1536)
```

**The equation:**
$$f = [v'; t'] \in \mathbb{R}^{1536}$$

Where:
- $v' = \text{CrossAttn}(v, t, t)$ — visual features informed by text
- $t' = \text{CrossAttn}(t, v, v)$ — text features informed by vision
- $f$ = fused representation (1536-d)

**Why bidirectional?**
- Vision informs text understanding
- Text informs vision understanding
- Both directions capture the full interaction

---

## 6. Stage 4: Feature Concatenation & Projection

### Concatenation

We concatenate the original and attended features:

```python
fused = torch.cat([v, v_attended, t, t_attended], dim=-1)
# Shape: (batch_size, 768 * 4) = (batch_size, 3072)
```

**Why include originals?**
- Preserves information that might be lost in attention
- Acts as a "skip connection" for the fusion

### Projection to Hidden Dim

```python
fusion_projection = nn.Linear(3072, 512)
hidden = fusion_projection(fused)
hidden = nn.ReLU()(hidden)
hidden = nn.Dropout(0.5)(hidden)
```

**The equation:**
$$h = \text{Dropout}(\text{ReLU}(W_f \cdot f + b_f))$$

Where:
- $W_f$ = weight matrix (512 × 3072)
- $b_f$ = bias vector (512-d)
- $f$ = fused features (3072-d)
- $h$ = hidden representation (512-d)

**Why 512?**
- Smaller than 3072 → reduces parameters
- Large enough to capture complex patterns
- Standard "sweet spot" for hidden dimensions

**Why ReLU?**
$$\text{ReLU}(x) = \max(0, x)$$

- Introduces non-linearity (without it, the network is just linear regression)
- Computationally cheap
- Helps with vanishing gradients

**Why Dropout(0.5)?**
- Randomly sets 50% of neurons to zero during training
- Prevents overfitting (model can't rely on specific neurons)
- Forces the model to learn redundant representations

---

## 7. Stage 5: Multi-Task Prediction Heads

### The Setup

We have **9 different attributes** to predict:
1. **theme** (9 classes): Sale, Discount, New Product, etc.
2. **sentiment** (3 classes): Positive, Neutral, Negative
3. **emotion** (5 classes): Joy, Surprise, Fear, Sadness, Anger
4. **dominant_colour** (10 classes): Red, Blue, Green, etc.
5. **attention_score** (3 classes): High, Medium, Low
6. **trust_safety** (3 classes): High, Medium, Low
7. **target_audience** (6 classes): Young Adults, Parents, etc.
8. **predicted_ctr** (3 classes): High, Medium, Low
9. **likelihood_shares** (3 classes): High, Medium, Low

### Architecture

Each head is a **2-layer MLP** (Multi-Layer Perceptron):

```python
class PredictionHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**The equation for each head:**
$$y = W_2 \cdot \text{Dropout}(\text{ReLU}(W_1 \cdot h + b_1)) + b_2$$

Where:
- $h$ = shared hidden representation (512-d)
- $W_1$ = first layer weights (256 × 512)
- $b_1$ = first layer bias (256-d)
- $W_2$ = second layer weights (num_classes × 256)
- $b_2$ = second layer bias (num_classes-d)
- $y$ = logits (raw scores for each class)

### Why Separate Heads?

**Shared backbone + task-specific heads** is a powerful pattern:

```
Shared Hidden Representation (512-d)
    ↓
    ├──→ Theme Head → 9 logits
    ├──→ Sentiment Head → 3 logits
    ├──→ Emotion Head → 5 logits
    ├──→ Colour Head → 10 logits
    ├──→ Attention Head → 3 logits
    ├──→ Trust Head → 3 logits
    ├──→ Audience Head → 6 logits
    ├──→ CTR Head → 3 logits
    └──→ Shares Head → 3 logits
```

**Benefits:**
1. **Parameter sharing:** The backbone learns general features useful for all tasks
2. **Task-specific learning:** Each head specializes in its attribute
3. **Regularization:** Multi-task learning acts as regularization (prevents overfitting)
4. **Efficiency:** One forward pass produces all 9 predictions

### Logits to Probabilities

The raw outputs (logits) are converted to probabilities using **softmax**:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Where:
- $z_i$ = logit for class $i$
- $C$ = number of classes
- $p_i$ = probability of class $i$
- $\sum p_i = 1$

**Example:**
```
Logits for theme: [2.3, 0.5, -1.2, 4.1, 0.8, -0.3, 1.5, 0.2, -0.5]
Probabilities:    [0.21, 0.03, 0.01, 0.73, 0.05, 0.02, 0.09, 0.02, 0.01]
Prediction:       "Discount" (class 3, probability 0.73)
```

---

## 8. Stage 6: Loss Computation

### Cross-Entropy Loss

For each attribute, we use **cross-entropy loss**:

$$L = -\sum_{i=1}^{C} y_i \log(p_i)$$

Where:
- $y_i$ = 1 if class $i$ is the true class, else 0 (one-hot)
- $p_i$ = predicted probability for class $i$
- $C$ = number of classes

**Simplified (when $y$ is one-hot):**
$$L = -\log(p_{\text{true}})$$

**Intuition:** The loss is the negative log of the probability assigned to the correct class.
- If $p_{\text{true}} = 0.99$, loss = $-\log(0.99) = 0.01$ (very small)
- If $p_{\text{true}} = 0.01$, loss = $-\log(0.01) = 4.6$ (very large)

### Label Smoothing

Instead of hard targets (0 or 1), we use **soft targets**:

$$y_i^{\text{smooth}} = (1 - \epsilon) y_i + \frac{\epsilon}{C}$$

Where $\epsilon = 0.2$ (smoothing factor) and $C$ = number of classes.

**Example with $\epsilon = 0.2$, $C = 3$:**
```
Hard target:     [1, 0, 0]
Smoothed target: [0.93, 0.033, 0.033]
```

**Why label smoothing?**
- Prevents the model from becoming **overconfident**
- Improves generalization
- Acts as regularization

### Total Loss

The total loss is the **sum** of all 9 attribute losses:

$$L_{\text{total}} = \sum_{a=1}^{9} L_a$$

Where $L_a$ is the cross-entropy loss for attribute $a$.

**Why sum (not average)?**
- Gives equal weight to each attribute
- Prevents attributes with more classes from dominating

---

## 9. Stage 7: Backpropagation & Optimization

### The Goal

We want to **minimize** the total loss by adjusting the model's parameters (weights and biases).

### Gradient Descent

**The update rule:**
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where:
- $\theta_t$ = parameters at step $t$
- $\eta$ = learning rate (step size)
- $\nabla L(\theta_t)$ = gradient of loss with respect to parameters

**Intuition:**
- Compute the gradient (direction of steepest ascent)
- Move in the opposite direction (steepest descent)
- Repeat

### AdamW Optimizer

We use **AdamW** (Adam with Weight Decay), not basic gradient descent.

**Adam maintains two moving averages:**
1. **First moment** (mean of gradients): $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
2. **Second moment** (variance of gradients): $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

**Update rule:**
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Where:
- $\hat{m}_t = m_t / (1-\beta_1^t)$ (bias correction)
- $\hat{v}_t = v_t / (1-\beta_2^t)$ (bias correction)
- $\lambda$ = weight decay coefficient
- $\eta$ = learning rate

**Why AdamW?**
- **Adaptive learning rates:** Different parameters get different learning rates
- **Momentum:** Smooths out noisy gradients
- **Weight decay:** Prevents overfitting by penalizing large weights

### Differential Learning Rates

Different parts of the model get **different learning rates**:

```python
optimizer = AdamW([
    {'params': resnet50.parameters(), 'lr': 1.5e-5},  # Slow for pretrained
    {'params': distilbert.parameters(), 'lr': 1.5e-5},  # Slow for pretrained
    {'params': fusion.parameters(), 'lr': 2e-4},  # Fast for new layers
    {'params': heads.parameters(), 'lr': 2e-4},  # Fast for new layers
])
```

**Why?**
- **Pretrained models** (ResNet, DistilBERT) already know a lot. We update them slowly to avoid forgetting.
- **New layers** (fusion, heads) are randomly initialized. They need to learn fast.

### Learning Rate Schedule

We use **warmup + cosine annealing**:

```
Learning Rate
    ↑
    │     ╱╲
    │    ╱  ╲
    │   ╱    ╲
    │  ╱      ╲
    │ ╱        ╲
    │╱          ╲___
    └──────────────────→ Epochs
    Warmup  Cosine Decay
```

**Phase 1: Warmup (5 epochs)**
$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

Learning rate increases linearly from 0 to max.

**Why warmup?**
- Early gradients are noisy
- Large learning rates early on can destabilize training
- Warmup allows the model to "settle in"

**Phase 2: Cosine Annealing**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}} \pi))$$

Learning rate decreases following a cosine curve.

**Why cosine?**
- Smooth decay (no sudden drops)
- Allows fine-tuning at the end
- Empirically works better than step decay

### Backpropagation

**The chain rule:**
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$

PyTorch computes this **automatically** using **autograd**:

```python
loss = compute_loss(predictions, labels)
loss.backward()  # Compute all gradients
optimizer.step()  # Update parameters
optimizer.zero_grad()  # Clear gradients for next iteration
```

**The flow:**
```
Forward pass: Input → Predictions → Loss
    ↓
Backward pass: Loss → Gradients (via chain rule)
    ↓
Update: Parameters -= learning_rate × gradients
```

---

## 10. Stage 8: Inference (Prediction)

### The Process

At inference time (after training):

```python
model.eval()  # Set to evaluation mode (disables dropout)
with torch.no_grad():  # Don't compute gradients (saves memory)
    predictions = model(image, text_input_ids, text_attention_mask)
```

**Key differences from training:**
1. **No dropout:** All neurons are active
2. **No gradients:** Saves memory and computation
3. **BatchNorm uses running statistics:** Not batch statistics

### From Logits to Final Predictions

```python
# Get logits from all 9 heads
logits = model(image, text)  # Dictionary: {attr: logits}

# Convert to probabilities
probabilities = {attr: F.softmax(logits[attr], dim=-1) for attr in logits}

# Get predicted class (argmax)
predictions = {attr: torch.argmax(probabilities[attr], dim=-1) for attr in probabilities}

# Convert to labels
labels = {attr: label_maps[attr][predictions[attr].item()] for attr in predictions}
```

**Example output:**
```python
{
    "theme": "Discount",
    "sentiment": "Positive",
    "emotion": "Joy",
    "dominant_colour": "Red",
    "attention_score": "High",
    "trust_safety": "Medium",
    "target_audience": "Young Adults",
    "predicted_ctr": "High",
    "likelihood_shares": "Medium"
}
```

---

## 11. Complete Mathematical Walkthrough

Let's trace a single sample through the entire model:

### Input
- Image: $I \in \mathbb{R}^{3 \times 224 \times 224}$
- Text: $T = [t_1, t_2, ..., t_{256}]$ (token IDs)

### Step 1: Visual Encoding
$$v_{2048} = \text{ResNet50}(I) \in \mathbb{R}^{2048}$$

$$v_{768} = W_v v_{2048} + b_v \in \mathbb{R}^{768}$$

### Step 2: Text Encoding
$$T_{768} = \text{DistilBERT}(T) \in \mathbb{R}^{256 \times 768}$$

$$\alpha = \text{softmax}(W_a T_{768}) \in \mathbb{R}^{256}$$

$$t_{768} = \sum_{i=1}^{256} \alpha_i T_{768,i} \in \mathbb{R}^{768}$$

### Step 3: Cross-Modal Fusion
$$v' = \text{MultiHeadAttn}(Q=v_{768}, K=t_{768}, V=t_{768}) \in \mathbb{R}^{768}$$

$$t' = \text{MultiHeadAttn}(Q=t_{768}, K=v_{768}, V=v_{768}) \in \mathbb{R}^{768}$$

$$f = [v_{768}; v'; t_{768}; t'] \in \mathbb{R}^{3072}$$

### Step 4: Projection
$$h = \text{Dropout}(\text{ReLU}(W_f f + b_f)) \in \mathbb{R}^{512}$$

### Step 5: Prediction
For each attribute $a$:
$$z_a = W_{a,2} \cdot \text{Dropout}(\text{ReLU}(W_{a,1} h + b_{a,1})) + b_{a,2}$$

$$p_a = \text{softmax}(z_a)$$

### Step 6: Loss
$$L = \sum_{a=1}^{9} -\sum_{i=1}^{C_a} y_{a,i} \log(p_{a,i})$$

### Step 7: Backpropagation
$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

---

## 12. Why This Architecture Works

### 1. **Multi-Modal Learning**
- Vision and text provide **complementary information**
- Combining them is better than either alone
- Cross-attention captures their interaction

### 2. **Transfer Learning**
- ResNet-50 and DistilBERT are pretrained on massive datasets
- They already know about images and text
- We only need to fine-tune them for ads

### 3. **Multi-Task Learning**
- 9 attributes share a common backbone
- Learning multiple tasks together improves generalization
- Each head specializes while sharing knowledge

### 4. **Attention Mechanisms**
- Self-attention (in DistilBERT) captures long-range dependencies in text
- Cross-attention captures vision-text interactions
- Both are crucial for understanding ads

### 5. **Regularization**
- Dropout (0.5) prevents overfitting
- Label smoothing (0.2) prevents overconfidence
- Weight decay (0.01) penalizes large weights

### 6. **Careful Optimization**
- Differential learning rates respect pretrained models
- Warmup prevents early instability
- Cosine annealing allows fine convergence

---

## 🎓 Summary

The FG_MFN model works by:

1. **Extracting visual features** using ResNet-50 (2048-d → 768-d)
2. **Extracting text features** using DistilBERT with attention pooling (768-d)
3. **Fusing modalities** using bidirectional cross-modal attention (3072-d)
4. **Projecting** to a shared hidden space (512-d)
5. **Predicting 9 attributes** using task-specific MLP heads
6. **Training** with cross-entropy loss + label smoothing
7. **Optimizing** with AdamW + differential learning rates + cosine schedule

Every component is carefully designed and justified. The architecture balances **expressiveness** (can learn complex patterns) with **regularization** (won't overfit) and **efficiency** (fast inference).

---

## 📖 Further Reading

- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Transformer:** Vaswani et al., "Attention Is All You Need" (2017)
- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- **DistilBERT:** Sanh et al., "DistilBERT, a distilled version of BERT" (2019)
- **AdamW:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
- **Label Smoothing:** Szegedy et al., "Rethinking the Inception Architecture" (2016)
