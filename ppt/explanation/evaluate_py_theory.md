# Deep Explanation: `evaluate.py` — Theory & Fundamentals

> **Purpose:** Understand *why* this script exists, *what* every concept means, and *how* the pieces fit together — without getting lost in code details.

---

## 📚 Table of Contents

1. [What is Model Evaluation?](#1-what-is-model-evaluation)
2. [The Big Picture: Where This Script Fits](#2-the-big-picture-where-this-script-fits)
3. [Core Concept 1: Train vs. Test Split](#3-core-concept-1-train-vs-test-split)
4. [Core Concept 2: Batch Processing](#4-core-concept-2-batch-processing)
5. [Core Concept 3: GPU vs. CPU](#5-core-concept-3-gpu-vs-cpu)
6. [Core Concept 4: Model Checkpoints](#6-core-concept-4-model-checkpoints)
7. [Core Concept 5: Configuration Files (YAML)](#7-core-concept-5-configuration-files-yaml)
8. [Core Concept 6: Label Maps](#8-core-concept-6-label-maps)
9. [Core Concept 7: Image Preprocessing Pipeline](#9-core-concept-7-image-preprocessing-pipeline)
10. [Core Concept 8: Text Tokenization](#10-core-concept-8-text-tokenization)
11. [Core Concept 9: DataLoaders in PyTorch](#11-core-concept-9-dataloaders-in-pytorch)
12. [Core Concept 10: Evaluation Metrics](#12-core-concept-10-evaluation-metrics)
13. [Core Concept 11: Confusion Matrix](#13-core-concept-11-confusion-matrix)
14. [Core Concept 12: Macro vs. Weighted Averaging](#14-core-concept-12-macro-vs-weighted-averaging)
15. [Core Concept 13: Visualizations](#15-core-concept-13-visualizations)
16. [Core Concept 14: Clean Architecture](#16-core-concept-14-clean-architecture)
17. [Core Concept 15: The Factory Pattern](#17-core-concept-15-the-factory-pattern)
18. [Core Concept 16: Closures in Python](#18-core-concept-16-closures-in-python)
19. [Core Concept 17: Bug Documentation as Institutional Memory](#19-core-concept-17-bug-documentation-as-institutional-memory)
20. [The Complete Flow: Step by Step](#20-the-complete-flow-step-by-step)
21. [Why This Design Matters](#21-why-this-design-matters)

---

## 1. What is Model Evaluation?

**Definition:** Model evaluation is the process of measuring how well a trained machine learning model performs on **new, unseen data**.

**Why it matters:**
- A model that performs perfectly on training data but fails on new data is **useless**.
- Evaluation tells us if the model has actually *learned* or just *memorized*.
- It helps us compare different models and choose the best one.

**Analogy:**
Think of it like a student taking a practice test (training) vs. a final exam (evaluation). The final exam has questions the student has never seen before. If the student does well on the final exam, they've truly learned the material.

---

## 2. The Big Picture: Where This Script Fits

```
┌─────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING WORKFLOW                       │
│                                                              │
│   1. Collect Data                                            │
│         ↓                                                    │
│   2. Preprocess Data (clean, transform)                      │
│         ↓                                                    │
│   3. Train Model (scripts/train.py)                          │
│         ↓                                                    │
│   4. Save Model (saved_models/fg_mfn_best.pt)                │
│         ↓                                                    │
│   5. Evaluate Model (scripts/evaluate.py) ← YOU ARE HERE    │
│         ↓                                                    │
│   6. Deploy Model (REST API + Web App)                       │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Evaluation is the **quality gate** before deployment. If the model doesn't perform well here, we go back to step 3 and retrain with different settings.

---

## 3. Core Concept 1: Train vs. Test Split

**The Problem:**
If you test a model on the same data it was trained on, you'll get overly optimistic results. The model has already seen the answers!

**The Solution:**
Split your data into **three parts**:

```
┌─────────────────────────────────────────────────────┐
│              FULL DATASET (4,860 samples)            │
│                                                      │
│   ┌──────────────┬──────────────┬──────────────┐    │
│   │   TRAIN      │    VAL       │    TEST      │    │
│   │   70%        │    15%       │    15%       │    │
│   │   3,402      │    729       │    729       │    │
│   └──────────────┴──────────────┴──────────────┘    │
│                                                      │
│   Used to         Used to         Used to            │
│   teach the       tune            measure final      │
│   model           hyperparameters  performance       │
└─────────────────────────────────────────────────────┘
```

**Why three splits?**
- **Train:** The model learns from this data.
- **Validation (Val):** Used during training to tune hyperparameters (learning rate, batch size, etc.) and decide when to stop training (early stopping).
- **Test:** Used **only once** at the end to measure final performance. The model never sees this data during training.

**Analogy:**
- Train = homework (you practice and learn)
- Val = practice exams (you check your understanding)
- Test = final exam (measures what you truly learned)

---

## 4. Core Concept 2: Batch Processing

**The Problem:**
Processing one sample at a time is **slow**. Modern GPUs can process many samples in parallel.

**The Solution:**
Process samples in **batches** (groups of N samples at once).

```
Without Batching (slow):
   Sample 1 → GPU → Result 1
   Sample 2 → GPU → Result 2
   Sample 3 → GPU → Result 3
   ... (729 times for test set)

With Batching (fast):
   [Sample 1, 2, ..., 32] → GPU → [Result 1, 2, ..., 32]
   [Sample 33, 34, ..., 64] → GPU → [Result 33, 34, ..., 64]
   ... (23 times for test set with batch_size=32)
```

**Trade-offs:**
- **Small batch size** (e.g., 8): Less memory, but slower and noisier gradients
- **Large batch size** (e.g., 128): Faster, but uses more memory and may generalize worse
- **Sweet spot:** 32 or 64 for most tasks

**Why batching works:**
GPUs are designed for **parallel computation**. A single matrix multiplication on 32 samples is almost as fast as on 1 sample, but you get 32x the throughput.

---

## 5. Core Concept 3: GPU vs. CPU

**CPU (Central Processing Unit):**
- A few powerful cores (4-16)
- Good for sequential tasks
- Used for general computing

**GPU (Graphics Processing Unit):**
- Thousands of small cores (1000s)
- Good for parallel tasks
- Originally designed for graphics, now used for deep learning

**Speed comparison:**
```
CPU:  ~10 samples/second
GPU:  ~500 samples/second  (50x faster!)
```

**The code:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This checks if a GPU is available. If yes, use it. If no, fall back to CPU.

**Why GPUs are faster for deep learning:**
Deep learning is mostly **matrix multiplications**. GPUs can do thousands of multiplications simultaneously, while CPUs do them one at a time.

---

## 6. Core Concept 4: Model Checkpoints

**Definition:** A checkpoint is a **snapshot** of a model's state at a specific point during training.

**What's saved:**
```
checkpoint.pt
├── model weights (the learned parameters)
├── optimizer state (for resuming training)
├── epoch number (which epoch was this?)
├── best metric value (how good was this snapshot?)
└── config (what settings were used?)
```

**Why we need them:**
- Training takes hours/days. If it crashes, we don't want to start over.
- We save the **best** checkpoint (highest validation accuracy) for evaluation.
- We can resume training from a checkpoint.

**Analogy:**
Like saving your progress in a video game. If you die, you can reload from the last save point.

---

## 7. Core Concept 5: Configuration Files (YAML)

**Definition:** YAML (YAML Ain't Markup Language) is a human-readable format for configuration files.

**Example:**
```yaml
model:
  name: FG_MFN
  visual_encoder: resnet50
  text_encoder: distilbert-base-uncased

training:
  batch_size: 32
  learning_rate: 0.0002
  epochs: 50

dataset:
  root: /data/ads
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

**Why use config files?**
- **Separation of concerns:** Code stays the same, only config changes.
- **Experimentation:** Easy to try different settings without changing code.
- **Reproducibility:** Others can reproduce your results by using the same config.
- **Version control:** Config files can be tracked in Git.

**Analogy:**
Like a recipe. The cooking method (code) stays the same, but you can change ingredients (config) to make different dishes.

---

## 8. Core Concept 6: Label Maps

**Definition:** A label map converts between **human-readable labels** (strings) and **model-readable indices** (integers).

**Example:**
```python
label_maps = {
    "theme": {
        "Sale": 0,
        "Discount": 1,
        "New Product": 2,
        ...
    },
    "sentiment": {
        "Positive": 0,
        "Neutral": 1,
        "Negative": 2
    }
}
```

**Why we need them:**
- Models work with numbers, not strings.
- We need to convert model output (numbers) back to labels for human interpretation.
- We need to convert human labels (strings) to numbers for training.

**The flow:**
```
Human label: "Sale"
    ↓ (label_map["theme"]["Sale"])
Model index: 0
    ↓ (model prediction)
Model output: 0
    ↓ (reverse label_map)
Human label: "Sale"
```

---

## 9. Core Concept 7: Image Preprocessing Pipeline

**Definition:** A series of transformations applied to images before feeding them to the model.

**Typical steps:**
```
Original Image (variable size, RGB)
    ↓ Resize to 224×224
    ↓ Normalize pixel values to [0, 1]
    ↓ Standardize using ImageNet mean & std
    ↓ Convert to PyTorch tensor (C, H, W format)
Final Tensor (3, 224, 224)
```

**Why each step matters:**

1. **Resize:** Neural networks require fixed-size inputs. ResNet-50 expects 224×224.

2. **Normalize:** Pixel values range from 0-255. We divide by 255 to get 0-1. This helps the model train faster.

3. **Standardize:** We subtract the mean and divide by the standard deviation of the ImageNet dataset. This matches the distribution the pretrained model expects.

4. **Convert to tensor:** PyTorch uses a specific format: (Channels, Height, Width) instead of the typical (Height, Width, Channels).

**Critical rule:** The preprocessing at **evaluation time must match training time exactly**. If you trained with 224×224 images but evaluate with 512×512, the model will fail.

---

## 10. Core Concept 8: Text Tokenization

**Definition:** Converting text into numbers that a model can process.

**The process:**
```
"Buy 2 Get 1 Free!"
    ↓ Tokenize (split into words/subwords)
["buy", "2", "get", "1", "free", "!"]
    ↓ Convert to IDs (lookup in vocabulary)
[2378, 1016, 2053, 1015, 2489, 999]
    ↓ Add special tokens ([CLS] and [SEP])
[101, 2378, 1016, 2053, 1015, 2489, 999, 102]
    ↓ Pad/truncate to max_length (256)
[101, 2378, ..., 102, 0, 0, 0, ...]  (padded with zeros)
```

**Key concepts:**

- **Vocabulary:** A dictionary mapping tokens to IDs. DistilBERT has a vocabulary of ~30,000 tokens.
- **Special tokens:** `[CLS]` (start), `[SEP]` (end), `[PAD]` (padding), `[UNK]` (unknown).
- **Max length:** The maximum number of tokens. Longer texts are truncated, shorter ones are padded.
- **Attention mask:** Tells the model which tokens are real (1) and which are padding (0).

**Why max_length=256?**
- Most ad texts are short (< 50 words).
- 256 tokens is enough to capture the meaning.
- Longer sequences use more memory and computation.

---

## 11. Core Concept 9: DataLoaders in PyTorch

**Definition:** A DataLoader wraps a Dataset and provides batching, shuffling, and parallel loading.

**What it does:**
```
Dataset (729 samples)
    ↓ DataLoader (batch_size=32)
Batch 1: samples[0:32]
Batch 2: samples[32:64]
...
Batch 23: samples[704:729]  (smaller, last batch)
```

**Key parameters:**
- `batch_size`: How many samples per batch
- `shuffle`: Whether to randomize order (False for evaluation)
- `num_workers`: How many parallel processes to load data (speeds up loading)

**Why we use DataLoaders:**
- **Memory efficiency:** Don't load all data at once.
- **Parallelism:** Load batch N+1 while processing batch N.
- **Standardization:** Same interface for training and evaluation.

---

## 12. Core Concept 10: Evaluation Metrics

**Definition:** Numerical measures of model performance.

**Common metrics:**

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions
```
- Simple and intuitive.
- **Problem:** Misleading for imbalanced datasets.
- Example: If 95% of ads are "Positive", a model that always predicts "Positive" gets 95% accuracy but is useless.

### Precision
```
Precision = True Positives / (True Positives + False Positives)
```
- Of all the times the model predicted "Positive", how many were correct?
- High precision = few false alarms.

### Recall
```
Recall = True Positives / (True Positives + False Negatives)
```
- Of all the actual "Positive" ads, how many did the model catch?
- High recall = few missed positives.

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall.
- Balances both metrics.
- Range: 0 (worst) to 1 (best).

**Why multiple metrics?**
Each metric captures a different aspect of performance. A model can have high accuracy but low recall, or vice versa. Using multiple metrics gives a complete picture.

---

## 13. Core Concept 11: Confusion Matrix

**Definition:** A table showing where the model made correct and incorrect predictions.

**Example (3-class classification):**
```
                Predicted
              Cat  Dog  Bird
Actual  Cat  [ 45   3    2  ]   ← 45 correct, 5 wrong
        Dog  [  2  48    0  ]   ← 48 correct, 2 wrong
        Bird [  1   2   47  ]   ← 47 correct, 3 wrong
```

**How to read it:**
- **Diagonal:** Correct predictions (45, 48, 47)
- **Off-diagonal:** Mistakes (the model confused Cat with Dog 3 times, etc.)

**Why it's useful:**
- Shows **which classes** the model confuses.
- Helps identify if the model has a **bias** toward certain classes.
- Reveals **patterns** in errors (e.g., always confusing similar classes).

---

## 14. Core Concept 12: Macro vs. Weighted Averaging

**The Problem:**
When you have multiple classes, you need to combine per-class metrics into one number. There are two common ways:

### Macro Averaging
```
Macro Accuracy = (Accuracy_class_1 + Accuracy_class_2 + ... + Accuracy_class_N) / N
```
- **Treats all classes equally**, regardless of size.
- Good when you care about **rare classes** as much as common ones.

### Weighted Averaging
```
Weighted Accuracy = Σ (Accuracy_class_i × Support_class_i) / Total Samples
```
- **Weights by class frequency** (support = number of samples).
- Good when you care about **overall performance** on the dataset as it is.

**Example:**
```
Class A: 100 samples, 90% accuracy
Class B: 10 samples, 50% accuracy

Macro Accuracy   = (0.90 + 0.50) / 2 = 0.70
Weighted Accuracy = (0.90 × 100 + 0.50 × 10) / 110 = 0.86
```

**Which to use?**
- **Macro:** When class imbalance is a concern (e.g., medical diagnosis where missing a rare disease is bad).
- **Weighted:** When the dataset distribution matches the real-world distribution.

---

## 15. Core Concept 13: Visualizations

**Definition:** Graphical representations of metrics and results.

**Why visualize?**
- **Humans understand images faster than numbers.**
- **Patterns are easier to see** in a chart than in a table.
- **Communication:** Easier to explain results to non-technical stakeholders.

**Common visualizations for evaluation:**

1. **Confusion Matrix Heatmap:** Shows where the model makes mistakes.
2. **Per-Attribute Bar Chart:** Compares accuracy across different attributes.
3. **Macro vs. Weighted Comparison:** Shows the impact of class imbalance.
4. **ROC Curves:** Shows the trade-off between true positive rate and false positive rate.
5. **Precision-Recall Curves:** Shows the trade-off between precision and recall.

**Analogy:**
A picture is worth a thousand numbers. Visualizations turn a table of metrics into an intuitive story.

---

## 16. Core Concept 14: Clean Architecture

**Definition:** A software design pattern that separates code into layers with clear responsibilities.

**The 4 layers in this project:**
```
┌─────────────────────────────────────────────┐
│  scripts/  (Entry points, CLI)              │  ← User-facing
├─────────────────────────────────────────────┤
│  app/  (FastAPI routes, HTTP layer)         │  ← Interface
├─────────────────────────────────────────────┤
│  use_cases/  (Business logic)               │  ← Orchestration
├─────────────────────────────────────────────┤
│  lib/  (Core domain code)                   │  ← Pure logic
└─────────────────────────────────────────────┘
```

**Rules:**
- **Upper layers can call lower layers**, but not vice versa.
- **Lower layers don't know about upper layers.**
- Each layer has a **single responsibility**.

**Why this matters:**
- **Testability:** You can test `lib/` without starting the API.
- **Flexibility:** You can swap the API (FastAPI → Flask) without changing business logic.
- **Maintainability:** Changes are localized to one layer.

**Analogy:**
Like a restaurant:
- **scripts/** = the menu (what customers see)
- **app/** = the waiter (takes orders)
- **use_cases/** = the kitchen manager (coordinates)
- **lib/** = the cooks (do the actual work)

---

## 17. Core Concept 15: The Factory Pattern

**Definition:** A design pattern where a function creates objects based on parameters, hiding the construction logic.

**Example:**
```python
def load_model(config, device, checkpoint_path):
    if config["model_type"] == "FG_MFN":
        return FG_MFN(config, device, checkpoint_path)
    elif config["model_type"] == "ResNet":
        return ResNet(config, device, checkpoint_path)
    # ... more model types
```

**Why use it?**
- **Decoupling:** The caller doesn't need to know how to construct each model.
- **Extensibility:** Add new models without changing the caller.
- **Centralization:** All construction logic is in one place.

**Analogy:**
Like ordering from a menu. You say "I want a burger" (call the factory), and the kitchen decides how to make it. You don't need to know the recipe.

---

## 18. Core Concept 16: Closures in Python

**Definition:** A closure is a function that remembers variables from the scope where it was created, even after that scope has finished executing.

**Example:**
```python
def outer(max_length):
    def inner(text):
        return tokenize(text, max_length=max_length)  # Remembers max_length
    return inner

tokenizer_fn = outer(max_length=256)
result = tokenizer_fn("Hello world")  # Uses max_length=256 automatically
```

**Why use closures here?**
- The dataset code calls `tokenizer_fn(text)` with just one argument.
- But `tokenize_text()` needs `max_length` and `tokenizer`.
- The closure **binds** these extra parameters so the dataset doesn't need to know about them.

**Analogy:**
Like a coffee order with saved preferences. You say "my usual" (call the function), and the barista knows your size, sugar, etc. (closure variables).

---

## 19. Core Concept 17: Bug Documentation as Institutional Memory

**The Pattern:**
Every script has comments like:
```python
# BUG-10 FIX: text_max_length now defaults to 256 (matching training config)
#             instead of the previous hardcoded 128.
```

**Why this matters:**
- **Context:** Future developers (and AI) understand *why* the code is the way it is.
- **Prevention:** Avoids re-introducing the same bug.
- **Learning:** New team members can learn from past mistakes.

**Analogy:**
Like warning signs on a road: "Danger: sharp curve ahead." They tell you what happened before and how to avoid it.

---

## 20. The Complete Flow: Step by Step

Here's what happens when you run:
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint saved_models/fg_mfn_best.pt
```

```
Step 1: Parse Arguments
   ↓ (--config, --checkpoint, --split, --output, --batch-size)

Step 2: Load Configuration
   ↓ (Read YAML file → Python dictionary)

Step 3: Setup Device
   ↓ (GPU if available, else CPU)

Step 4: Load Model
   ↓ (Create architecture → Load weights → Move to device → Set eval mode)

Step 5: Load Label Maps
   ↓ (String ↔ Integer conversions)

Step 6: Build Image Pipeline
   ↓ (Resize → Normalize → Standardize → Tensor)

Step 7: Build Text Pipeline (if use_text=True)
   ↓ (Clean text → Tokenize → Pad/Truncate)

Step 8: Load Dataset
   ↓ (Read images + text from disk → Apply pipelines)

Step 9: Create DataLoader
   ↓ (Batch the dataset)

Step 10: Run Evaluation
   ↓ (For each batch: forward pass → collect predictions)

Step 11: Compute Metrics
   ↓ (Accuracy, F1, Precision, Recall, Confusion Matrix, ...)

Step 12: Save Results
   ↓ (Write JSON + CSV files)

Step 13: Print Summary
   ↓ (Show key metrics to console)

Step 14: Generate Visualizations
   ↓ (Create PNG plots)
```

---

## 21. Why This Design Matters

### 1. **Reproducibility**
Anyone can run the same evaluation and get the same results by using the same config and checkpoint.

### 2. **Modularity**
Each component (model, dataset, metrics, viz) can be swapped or upgraded independently.

### 3. **Debugging**
When something goes wrong, you can isolate the problem:
- Bad metrics? → Check the model checkpoint.
- Crashes? → Check the data pipeline.
- Wrong numbers? → Check the config.

### 4. **Scalability**
The same script works whether you have 100 or 100,000 test samples.

### 5. **Communication**
Visualizations and metrics make it easy to explain results to others.

---

## 🎓 Key Takeaways

1. **Evaluation is the quality gate** before deployment.
2. **Train/val/test splits** prevent overfitting and give honest performance estimates.
3. **Batching** enables efficient GPU utilization.
4. **Preprocessing consistency** between training and evaluation is critical.
5. **Multiple metrics** give a complete picture of performance.
6. **Clean architecture** makes code testable, maintainable, and flexible.
7. **Documentation of bugs** preserves institutional knowledge.

---

## 📖 Further Reading

- **PyTorch Documentation:** [pytorch.org/docs](https://pytorch.org/docs/)
- **scikit-learn Metrics:** [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **HuggingFace Tokenizers:** [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- **Clean Architecture:** Robert C. Martin's book
- **Design Patterns:** Gang of Four book (Factory, Strategy, etc.)

---

**Next:** Read the actual code in `evaluate.py` with this theory in mind, and everything will click into place.
