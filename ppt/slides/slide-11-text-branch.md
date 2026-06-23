# Slide 11 — FG_MFN Model (Text Branch)

## What to Say (Speaker Notes)
"Now let's look at the **text branch**. The text branch is responsible for understanding the **words inside the ad image**. First, an OCR engine extracts the raw text from the image. Then I clean up common OCR artifacts (like extra spaces or misread characters). The cleaned text is passed through **DistilBERT**, which is a smaller, faster version of Google's famous BERT model. DistilBERT is **pre-trained** on a huge corpus of English text, so it already understands grammar, synonyms, and context. I use the **base, uncased** version from HuggingFace. The text is tokenized into sub-word pieces with a **maximum length of 256 tokens**. I use **attention-weighted pooling** — instead of just taking the last hidden state, I weight each token by its attention score, so important words contribute more. The output is a **768-dimensional feature vector** for each text."

## What to Show on Screen

```
📝 TEXT BRANCH — DistilBERT

   OCR Text  →  Clean  →  Tokenize  →  DistilBERT
   "Buy now!"    "Buy now!"   [CLS] buy now [SEP]
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │   DistilBERT         │
                              │   (base, uncased)    │
                              │   pretrained on      │
                              │   English corpus     │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              Text Feature Vector (768-d)

   ⚙️  KEY SETTINGS:
      • Model: DistilBERT (base, uncased) — HuggingFace
      • Pooling: attention-weighted (not just last hidden state)
      • Max token length: 256
      • Output: 768-d feature per text
```

## Visual Suggestion
- Show the **tokenization step** visually — words broken into sub-word pieces.
- Add a small **HuggingFace logo** to indicate where the model comes from.
- Highlight the **768-d output** with a colored box.

## Key Talking Points
- DistilBERT is **6× faster** than BERT but retains **97% of its performance** — perfect for our use case.
- **Attention-weighted pooling** is a small but important detail — it gives better results than naive pooling.
- The text branch only sees **what OCR extracted** — so OCR quality matters (more on that later).
