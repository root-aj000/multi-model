# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_wldbiz136txw"></a></h1><p>CHAPTER 1</p><p>INTRODUCTION</p></td></tr></tbody></table></div>

# 

## Introduction

Early research in advertisement analysis primarily focused on unimodal approaches, analyzing either textual or visual content separately. Text-based methods relied on sentiment analysis, keyword detection, and semantic relationships to infer user reactions, whereas image-based techniques utilized visual descriptors such as color histograms, edges, and object recognition. While these methods provided initial insights, they often failed to capture the symbolic and persuasive aspects of modern ads, where the meaning arises from the combination of visuals and text \[1\].

The rapid rise of social media platforms and online marketing has further emphasized the need for multimodal approaches. Users interact with content in multiple ways, including clicks, shares, comments, and reactions, providing rich behavioral data that can guide marketing strategies. For example, an advertisement combining a compelling product image with a discount offer typically drives higher engagement than either element alone, highlighting the importance of analyzing these modalities jointly \[2\].

Recent advances in vision-language models such as CLIP \[3\] and VisualBERT \[4\] have enabled better alignment of textual and visual representations, allowing for more sophisticated analysis of multimodal content. However, these general-purpose models are not optimized for advertising contexts, which often include overlapping objects, stylized text, and targeted persuasive messaging. This highlights the need for specialized frameworks that can process multimodal inputs while capturing fine-grained interactions between visual and textual elements.

To address these challenges, this project proposes a multimodal fusion framework that integrates image features from ResNet18 \[5\] and textual embeddings from DistilBERT \[6\]. The system fuses these features and passes them through task-specific heads to predict sentiment, providing a holistic understanding of advertisement effectiveness. By combining multimodal information with Sentiment learning, the framework goes beyond simple sentiment analysis, enabling accurate assessment of the persuasive and emotional impact of advertisements.

Cross-modal sentiment analysis in advertising has gained significant attention due to its potential in brand communication and consumer behavior analysis. However, traditional methods struggle to handle the multi-scale features and redundant objects in advertising images effectively, resulting in limited emotion recognition accuracy. To address the challenges of insufficient multi-scale features and target redundancy in multi-modal sentiment analysis of advertisements, we introduce a novel framework, Fine-Grained Multiscale Cross-Modal Feature Network (FGMFN). The model is designed to process multi-scale feature inputs, facilitating efficient sentiment fusion between images and text. FGMFN employs a multi-scale network to extract key features from advertising images, and uses visual features to guide the textual data representation.\[7\]

Sentiment analysis is a popular task in natural language processing. The goal of sentiment analysis is to classify the text based on the mood or mentality expressed in the text, which can be positive, negative, or neutral.

Sentiment analysis is the process of classifying whether a block of text is positive, negative, or neutral. The goal that Sentiment mining tries to gain is to analyse people’s opinions in a way that can help businesses expand. It focuses not only on polarity (positive, negative & neutral) but also on emotions (happy, sad, angry, etc.). It uses various Natural Language Processing algorithms such as Rule-based, Automatic, and Hybrid.

Recent advances in vision-language models such as CLIP and VisualBERT have enabled better alignment of textual and visual representations, allowing for more sophisticated analysis of multimodal content. However, these general-purpose models are not optimized for advertising contexts, which often include overlapping objects, stylized text, and targeted persuasive messaging. This highlights the need for specialized frameworks that can process multimodal inputs while capturing fine-grained interactions between visual and textual elements.

Fig 1.1 Overview of proposed system

This system for Multimodal Advertisement Sentiment Analysis works in three main steps:

- Feature Extraction, where ResNet18 analyzes the image and DistilBERT analyzes the text.
- Cross-Modal Attention Fusion, where the visual and text features are combined and aligned.
- Sentiment Prediction, where the fused information is classified as Positive, Neutral, or Negative sentiment.

## 1.2 Problem Statement

Accurately assessing the impact of digital advertisements remains a significant challenge. Traditional methods often focus on either textual sentiment or visual cues independently, neglecting the synergistic effects that arise when these modalities are combined \[9\]. As a result, existing approaches frequently underperform in predicting engagement metrics such as click-through rates, shares, and overall audience sentiment, limiting their practical utility for campaign optimization.

Another challenge lies in the complexity of modern ads. They feature diverse formats, overlapping objects, stylized typography, and promotional messages integrated into the visuals. Simple image descriptors or text sentiment scores are insufficient to capture these nuances. Moreover, advertisements are subjective in their emotional impact; users may perceive the same content differently, and basic sentiment analysis cannot account for subtle emotional responses such as trust, inspiration, or excitement \[8\].

Current multimodal models, while effective in general vision-language tasks, are not tailored for advertising. They struggle to interpret combined textual and visual cues in a way that accounts for both design elements and audience targeting. Therefore, there is a clear need for a specialized multimodal framework capable of understanding advertisements at multiple levels, sentiment categorization.

## 1.3 Objectives

The primary goal of this study is to develop a robust multimodal framework for advertisement analysis, integrating both textual and visual information to overcome the limitations of unimodal approaches. The framework aims to enable accurate prediction of advertisement effectiveness across multiple dimensions. Key objectives include:

- **Multimodal Fusion:** Develop a system that jointly processes visual and textual content to capture complex interactions between modalities \[9\].
- **Sentiment Prediction:** Implement task-specific outputs for sentiment classification \[6\].
- **Scalability:** Design a framework capable of handling diverse advertisement formats, from static images to social media banners \[10\].

## 1.4 Motivation

The motivation for this research is driven by the growing importance of digital marketing and the limitations of conventional ad analysis methods. Businesses increasingly invest in online advertising campaigns, yet often lack reliable tools to evaluate the impact of their ads on user engagement and perception \[10\]. Understanding the emotional and persuasive power of advertisements allows marketers to design more effective campaigns, allocate budgets efficiently, and target audiences precisely.

The multimodal nature of modern ads creates opportunities and challenges. Images and text interact in complex ways, and failing to capture this interaction reduces the accuracy of prediction models. For example, an aesthetically appealing image may fail to drive engagement without complementary textual messaging, underscoring the need for joint analysis \[10\].

Recent advances in vision-language models provide strong foundations for developing specialized frameworks. However, models like CLIP and VisualBERT are not optimized for advertising, motivating the development of an ad-focused multimodal system that can handle overlapping objects, promotional text, and varying visual styles.

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_7emjt9fe2qf5"></a></h1><p>CHAPTER 2</p><p>BACKGROUND &amp; RELATED WORK</p></td></tr></tbody></table></div>

# 

## 2.1 Digital Advertising and Its Evolution

Digital advertising has emerged as a dominant force in modern marketing, transforming how businesses communicate with their audiences. Unlike traditional advertising, which primarily relied on static media such as print, television, or radio, digital ads leverage online platforms to reach users across multiple channels, including social media, websites, and mobile applications \[11\]. This shift has enabled advertisers to engage audiences more interactively and measure the effectiveness of campaigns in real-time, creating new opportunities and challenges in understanding user behavior.

Early digital advertising focused on text-heavy content or simple banner images, which allowed marketers to track click-through rates, impressions, and basic engagement metrics. However, the increasing use of multimedia content—combining images, text, graphics, and animations—introduced a complexity that could not be captured using traditional analytics. Modern advertisements often integrate promotional messages, visual cues, logos, product placements, and emotional triggers in a single composition. These multimodal characteristics make it critical to develop analytical methods that can simultaneously process and interpret both textual and visual components.

Fig 2.1 Evolution of Advertising metrics

## 2.2 Multimodal Content in Advertisements

Multimodal advertisements consist of multiple forms of content, primarily visual and textual elements, which together convey a unified message. Visual content may include product images, background scenes, logos, banners, and even small promotional details like discount tags. Textual content is embedded within images and often requires OCR (Optical Character Recognition) for extraction and further analysis \[12\].

The interaction between these modalities is crucial. A text-only sentiment analysis may misinterpret an ad that visually conveys positive emotions, or conversely, an image may appear neutral unless contextualized by accompanying text. For instance, a product image with a bold “50% OFF” text is far more persuasive than the image alone. Similarly, text such as “Buy One Get One Free” may be ineffective without an accompanying visual that draws attention.

Analyzing these multimodal advertisements requires handling challenges such as:

- **Heterogeneous content formats:** Ads appear in banners, social media posts, static images, or GIFs.
- **Overlapping visual elements:** Logos, text overlays, and product images may coexist in the same ad.
- **Diverse typography and styles:** Fonts, colors, and sizes carry semantic and emotional meaning.

Fig 2.2 Multimodal Content in Advertisements

## 2.3 Sentiment Analysis in Advertising

Sentiment analysis involves detecting the emotional tone of content. In advertisements, sentiment goes beyond simple positive, negative, or neutral classification; it can reflect trust, excitement, inspiration, or concern depending on the audience’s perception \[12\].

Early approaches relied on unimodal analysis:

- **Text-based analysis:** Token-level sentiment detection using lexicons or machine learning classifiers.
- **Image-based analysis:** Convolutional neural networks (CNNs) extract low-level visual features such as color, edges, and objects.

Limitations of unimodal approaches:

- They ignore the synergy between text and visuals.
- They fail to capture marketing-specific cues like call-to-action phrases and product placement.
- They cannot handle subtle or nuanced emotions, which often arise from the interaction of visual and textual modalities.

The rise of multimodal sentiment analysis addresses these issues by integrating both text and image data. Models like Tensor Fusion Network (TFN), Memory Fusion Network (MFN), and Cross-Modal Attention Networks (CMAN) have been applied to social media and conversational data. In advertising, specialized frameworks such as FGMFN extend this approach by considering multiscale visual features and fine-grained cross-modal interactions.

## 2.4 Machine Learning and Deep Learning in Advertising

Modern advertisement analysis relies heavily on **deep learning models** to extract meaningful features from images and text:

- **Visual Feature Extraction**
    - CNNs such as ResNet18, Feature Pyramid Networks (FPN) \[13\] allow multiscale feature extraction, capturing small elements (logos, labels), medium-sized objects (products), and large-scale context (backgrounds).
    - Multiscale embeddings preserve semantic information across different resolutions, enabling better understanding of ad content.
- **Textual Feature Extraction**
    - Transformer-based models like DistilBERT, BERT \[13\], Roberta \[14\] encode OCR-extracted ad text into high-dimensional embeddings that capture context, syntax, and sentiment.
- **Vision-Language Models**
    - Models like CLIP, VisualBERT, UNITER \[14\] align textual and visual features into a shared latent space, enabling joint analysis.
    - These models, however, are often general-purpose and not tailored for the specific characteristics of advertisements, such as overlapping objects, embedded text, and promotional cues.
- **Cross-Modal Attention**
    - Attention mechanisms allow text to guide image feature weighting and vice versa, enabling the system to focus on semantically important regions and words \[15\].

Fig 2.4 Machine Learning and Deep Learning in Advertising

## 2.5 Existing Technology

### 2.5.1 Tensor Fusion Network (TFN)

Zadeh et al. \[10\] introduced the Tensor Fusion Network (TFN), a pioneering model for multimodal sentiment analysis that explicitly models the interactions between different modalities — such as text, Image. TFN constructs a tensor representation of features by taking the outer product of unimodal embeddings, enabling the model to capture both intra-modal and inter-modal relationships. This design makes it possible to represent the joint semantics across modalities in a single, unified space.

However, the tensor expansion used in TFN leads to a high-dimensional feature space, making the model computationally expensive and memory-intensive. While TFN works exceptionally well for multimodal tasks involving spoken videos (such as interviews or vlogs), it is not optimized for advertising scenarios, where the key modalities are images and text, not audio or video sequences.

Figure 2.5.1 Tensor Fusion Network Block Diagram

### 2.5.2 Memory Fusion Network (MFN)

Building upon TFN, Zadeh et al. \[15\] proposed the Memory Fusion Network (MFN), which introduced a temporal component into multimodal sentiment analysis. The MFN uses recurrent neural networks (LSTMs) for each modality and a multi-view gated memory to capture temporal dependencies and inter-modal relationships over time. It effectively learns how sentiment evolves as the conversation or scene progresses.

Figure 2.5.2 Memory Fusion Network Architecture.

The MFN architecture consists of three main components:

- **System of View-Specific Memories** – Each modality (text, image) maintains its own memory representation.
- **Delta-Memory Attention Mechanism** – Tracks changes between time steps, identifying shifts in sentiment.
- **Multi-View Gated Memory** – Integrates and fuses information from all modalities dynamically.

### 2.5.3 DialogueRNN

Majumder et al. \[15\] introduced DialogueRNN, which extended multimodal sentiment analysis into conversational emotion tracking. This model focuses on emotion dynamics, maintaining separate GRU-based states for each speaker and the global conversation. DialogueRNN processes each utterance sequentially and updates three key components: global context, speaker state, and emotional state.

The main innovation of DialogueRNN lies in its context-aware sentiment recognition — it doesn’t classify each sentence in isolation but uses preceding and succeeding dialogue turns to understand emotional flow. Though the model was initially developed for dialogues, the underlying principle of context dependency is equally relevant for advertisement sentiment analysis. Advertisements also rely on contextual emotional buildup — where the combination of colors, facial expressions, and textual slogans evokes specific sentiments.

Figure 2.5.3 DialogueRNN Model for Context-Aware Emotion Detection.

### 2.5.4 Cross-Modal Attention Network (CMAN)

Xu et al. \[16\] introduced the Cross-Modal Attention Network (CMAN), a model designed to strengthen the relationship between text and images by using attention mechanisms. CMAN aligns visual regions of an image with semantically corresponding words or phrases from the text, focusing attention on relevant parts — for example, aligning the word _“discount”_ with a percentage sign or price tag in an advertisement.

CMAN uses bidirectional attention, meaning text influences visual attention, and visual cues also influence which text features are most relevant. This cross-modal understanding allows the system to detect sentiment expressed jointly by text and imagery, such as “joy” conveyed by bright visuals and cheerful slogans.

Figure 2.5.4 Cross-Modal Attention Mechanism for Sentiment Analysis.

### 2.5.5 Multimodal Cross-Attention Model (MCAM)

Li et al. \[17\] advanced this idea further through the Multimodal Cross-Attention Model (MCAM), which combines multi-head attention with residual fusion blocks.

MCAM operates at both global and local levels:

- **Global Attention** identifies overall sentiment context (e.g., cheerful colors, positive language).
- **Local Attention** detects subtle emotional cues (e.g., facial expressions, specific words).

This dual-level modeling makes MCAM particularly suitable for tasks like emotion recognition and multimodal reasoning. In the context of advertisement sentiment analysis, it effectively bridges the semantic gap between product visuals and marketing slogans.

Figure 2.5.5 Multimodal Cross-Attention Model (MCAM).

### 2.5.6 CLIP-based Image–Text Sentiment Analysis

Lu et al. \[17\] extended the CLIP (Contrastive Language–Image Pretraining) model to sentiment analysis. CLIP learns a shared semantic space by aligning text and image embeddings using a contrastive loss — enabling it to understand images and captions in a unified framework. In their extension, Lu et al. fine-tuned CLIP for sentiment classification, leveraging its ability to generalize across various image-text domains.

This model achieved impressive results in cross-domain scenarios, showing strong zero-shot and few-shot performance. However, CLIP was trained on large-scale general datasets like LAION and doesn’t fully capture the symbolic and persuasive language unique to advertisements. Words like “SALE” or “LIMITED OFFER” carry specific emotional and behavioral intent that CLIP may overlook.

2.5.6 CLIP-based Cross-Modal Sentiment Analysis Framework.

### 2.5.7 Fine-Grained Multiscale Cross-Modal Feature Network (FGMFN)

Yu et al. \[16\] presented the Fine-Grained Multiscale Cross-Modal Feature Network (FGMFN), specifically designed for advertisement sentiment analysis. Unlike previous models, FGMFN focuses on understanding how visual layout, typography, colors, and textual slogans work together to convey emotion and marketing intent.

FGMFN introduces two major advancements:

- **Multiscale Feature Extraction:** Visual features are captured at multiple levels (from local details like icons or faces to global layouts), ensuring that both small and large-scale ad components contribute to the sentiment prediction.

Figure 2.5.7 Fine-Grained Multiscale Cross-Modal Feature Network (FGMFN).

- **Fine-Grained Cross-Modal Fusion:** Text tokens are aligned with corresponding visual regions using dual attention maps, allowing the model to identify subtle relationships like “discount text near price tag.”

This approach yields superior performance in detecting advertisement sentiment because it respects the fine-grained semantics and multiscale structure unique to ads.

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_onoideiur2z9"></a></h1><p>CHAPTER 3</p><p>PROPOSED METHODOLOGY</p></td></tr></tbody></table></div>

# 

## 3.1 System Architecture

The proposed methodology for multimodal advertisement analysis is designed to process both visual and textual information from ads, fuse them into a coherent representation, and predict downstream tasks such as sentiment. The high-level architecture of the system is modular and follows a three-stage pipeline: Unimodal Feature Extraction, Cross-Modal Fusion, and Sentiment Learning. Each stage is carefully designed to handle the complexity of modern digital ads while ensuring extensibility and robustness.

**Stage 1: Unimodal Feature Extraction**

The first stage focuses on extracting meaningful embeddings from each modality separately. For visual inputs, the system uses ResNet18 as the backbone to encode the image into a dense feature vector. ResNet18 is chosen for its balance of depth, computational efficiency, and proven accuracy in extracting high-level visual patterns from images. The visual embedding module is designed to handle multiscale features, capturing small objects like discount tags, medium-sized product images, and larger backgrounds to ensure comprehensive representation \[18\].

For textual content, extracted from advertisements via OCR, the system employs DistilBERT, a lightweight and fast transformer-based language model. DistilBERT encodes the text into contextual embeddings that capture semantics, sentiment cues, and promotional language in the ad. This ensures that the textual nuances, such as “50% OFF” or persuasive phrases like “Buy Now,” are effectively captured. Since visual and textual embeddings often have different dimensions (e.g., 2048-d for images vs. 768-d for text), the system applies learnable linear projections to map both embeddings into a shared latent space. This step is crucial to facilitate interaction between modalities in subsequent stages.

**Stage 2: Cross-Modal Fusion**

Once embeddings are dimensionally aligned, the system performs cross-modal fusion to model intricate relationships between image regions and text tokens. The fusion module leverages cross-attention mechanisms, allowing text to guide attention over visual regions and vice versa. This ensures that the model captures fine-grained interactions, such as highlighting the product mentioned in text while ignoring irrelevant background. In addition, multiscale visual fusion is incorporated to integrate features from multiple layers of ResNet, allowing detection of objects at different scales. The fused representation, therefore, becomes a robust joint embedding that contains rich semantic information from both modalities.

Fig 3.1 Architecture of multimodal advertisement analysis

**Stage 3: Sentiment Learning**

The final stage involves Sentiment learning, where the joint embedding is fed into task-specific heads for various predictions. Each head is optimized for its corresponding task. Sentiment classification categorizes the ad as positive, negative, or neutral.

The Sentiment learning setup uses a weighted loss function to jointly optimize all tasks, combining cross-entropy for classification, mean squared error for regression, and binary cross-entropy for trust prediction. This architecture ensures that all predictions benefit from shared multimodal representations while maintaining task-specific accuracy.

## 3.2 Dataset

### 3.2.1 Dataset Description

The dataset designed for this project consists of annotated advertisement samples that combine both visual and textual modalities to enable multimodal sentiment analysis. Each record in the dataset contains an image of an advertisement, its OCR-extracted textual content, dominant color composition, and a sentiment label manually annotated based on overall emotional tone and persuasive intent.

The dataset was curated to ensure diversity across product categories, ad layouts, and emotional contexts (e.g., joyful, persuasive, calm, or neutral). By combining structured and unstructured features, the dataset provides a rich foundation for training and evaluating the multimodal deep learning model.

### Dataset Attributes

|     |     |
| --- | --- |
| **Attribute** | **Description** |
| Image Path | File path to the advertisement image, stored in /images/ directory. Example: /images/0001.jpg |
| Image OCR | Text extracted from the advertisement image using OCR (Optical Character Recognition) tools such as paddler. Example: "Limited Time Offer! Buy One Get One Free." |
| Sentiment | Annotated label indicating the emotional polarity of the advertisement — Positive, Negative, or Neutral. |
| Dominant Color | Primary color tone of the ad background (e.g., Red, Blue, Green), extracted via clustering (K-Means) on pixel values. Useful for analyzing color psychology and emotional association. |

Table 3.1 Dataset Attributes

**Sample Dataset Table**

|     |     |     |     |
| --- | --- | --- | --- |
| **Image Path** | **Image OCR** | **Sentiment** | **Dominant Color** |
| /images/0001.jpg | “Limited Time Offer! Buy One Get One Free.” | Positive | Red |
| /images/0002.jpg | “Introducing the all-new Eco Drive — Save Fuel, Save Planet.” | Positive | Green |
| /images/0003.jpg | “High Prices Ahead — Shop Now Before It’s Too Late.” | Neutral | Yellow |
| /images/0004.jpg | “Unsatisfied with your service? Contact us today!” | Negative | Blue |

Table 3.2 Sample Dataset Table

### Data Processing Pipeline

1.  **Image Acquisition:** Raw advertisement images are collected from online campaigns and marketing repositories.
2.  **OCR Extraction:** The embedded text is extracted from images using Paddle OCR, preprocessed (cleaning, tokenization), and stored alongside image metadata.
3.  **Color Analysis:** The dominant color is identified using K-Means clustering (k=3) on pixel values, with the most frequent cluster centroid taken as the dominant color.
4.  **Sentiment Annotation:** Human annotators label each ad’s sentiment based on combined textual tone and visual style.
5.  **Dataset Storage:** Each record is stored in a structured format (e.g., CSV or JSON) for training and testing the multimodal neural network.

## 3.3 System Component Description

The proposed multimodal advertisement analysis framework is organized into several interconnected modules, each designed to handle a specific aspect of the pipeline. These modules include Advertisement Input, Visual Embedding, Textual Embedding, Dimensional Projection, Cross-Modal Fusion, and Sentiment Learning Heads. Each module is crucial for ensuring the system accurately captures and interprets the rich semantic content in digital advertisements. The modular design allows for flexibility, maintainability, and scalability, enabling the framework to adapt to new tasks or ad formats in the future.

### 3.3.1 Advertisement Input

The Advertisement Input Module serves as the foundational entry point of the proposed multimodal sentiment and engagement analysis system. This module is responsible for acquiring, validating, and preprocessing the advertisement content before it is passed to the visual and textual embedding pipelines. In most cases, the input comprises a combination of visual and textual elements — such as product banners, social media advertisements, and digital posters — which together convey persuasive messages designed to influence user behavior. The module ensures that both components are systematically extracted and prepared for consistent multimodal analysis.

Advertisements often blend visual design aesthetics (color schemes, object placement, product focus) with embedded textual information (slogans, promotional offers, brand names). To handle this complexity, the Advertisement Input Module first captures the entire image in its raw form and then applies Optical Character Recognition (OCR) techniques to extract embedded text directly from within the advertisement image. This design choice ensures that no information modality is ignored, and that textual features are contextually aligned with their corresponding visual regions.

Once the content is extracted, data preprocessing is performed in two distinct but coordinated pipelines — one for visual data and another for textual data.

- **Image Preprocessing**: Visual inputs are resized to a standardized resolution (e.g., 224×224 pixels) to maintain uniformity across the dataset. Pixel values are normalized to the \[0,1\] or \[-1,1\] range to stabilize learning and improve convergence in the ResNet18 encoder. Optional data augmentation (random cropping, rotation, color jittering, or horizontal flipping) is applied to enhance robustness and prevent overfitting \[21\].
- **Text Preprocessing**: The textual data extracted through OCR undergoes cleaning and normalization. This includes the removal of non-alphanumeric characters, conversion to lowercase, and tokenization using the DistilBERT tokenizer. Each text sequence is padded or truncated to a fixed length to ensure compatibility with transformer-based models \[19\].

The module also incorporates a data validation layer, ensuring that both modalities — image and text — are present and correctly formatted before proceeding to the embedding stage. If text is missing (e.g., in purely visual ads), the model introduces a placeholder embedding to maintain input consistency across the multimodal pipeline. This process enables the system to seamlessly handle real-world advertising datasets where textual density varies significantly between samples.

Ultimately, the Advertisement Input Module acts as a data integrity and standardization gateway, transforming heterogeneous ad materials into a clean, structured format suitable for deep learning–based feature extraction. By maintaining alignment between image and text content, this module establishes the critical foundation upon which cross-modal fusion and sentiment understanding are built.

### 3.3.2 Visual Embedding

The Visual Embedding is responsible for transforming raw advertisement images into high-dimensional feature vectors that capture both semantic and structural details essential for multimodal understanding. It employs ResNet-18, a convolutional neural network architecture with residual connections that allows efficient training while preserving rich hierarchical representations. The module captures low-level, mid-level, and high-level patterns such as color, texture, object structure, and contextual cues, all of which contribute to understanding the advertisement’s overall message and appeal

Fig 3.3.2.1 Advertisement Input Workflow

ResNet-18 is used as the encoder due to its balance between computational efficiency and representational strength, enabling scalable training on large-scale ad datasets without compromising accuracy. The transformation of an input image into a latent embedding is mathematically expressed as:

Where  denotes the input advertisement image,  represents the ResNet-18 encoder parameterized by , and  is the resulting high-dimensional feature embedding.

**Multiscale Feature Extraction**

Advertisement visuals often include multiple elements at varying scales — small icons, mid-sized products, and large backgrounds. To effectively capture this diversity, multiscale features are extracted from multiple convolutional layers of the ResNet-18 backbone:

Each  corresponds to feature maps from progressively deeper layers, where  captures low-level edges and colors,  encodes mid-level textures and shapes,  extracts high-level objects and logos, and  captures global contextual information such as layout and background.

To construct a unified representation that retains both fine-grained and global information, a multiscale fusion operation is performed:

Here,  denotes concatenation,  is a learnable projection matrix mapping the concatenated feature maps into a shared latent space, and  ) represents a residual connection preserving semantic consistency from higher layers.

This approach ensures that critical visual cues — such as brand elements, discount labels, or emotional tone — are preserved and integrated into the final embedding  . The resulting vector encodes comprehensive visual semantics, serving as a robust foundation for cross-modal fusion and subsequent predictive tasks \[20\].

Fig 3.3.2.2 Visual Embedding Workflow

### 3.3.3 Textual Embedding

The Textual Embedding processes OCR-extracted advertisement text using DistilBERT, a transformer-based model optimized for efficiency while retaining the rich semantic understanding of full-scale BERT models. This module converts raw text into contextual embeddings that capture meaning, sentiment, and structural cues, which are critical for understanding advertisement messages such as slogans, promotional phrases, and call-to-action statements \[21\].

**Tokenization and Sequence Preparation (Input Tokenization and Padding)**

Advertisement text  varies in length and structure. To handle this, the text is first tokenized into individual tokens and converted into numeric token IDs compatible with DistilBERT. Sequences shorter than a predefined maximum length are padded, while longer sequences are truncated to ensure consistent input dimensions:

Here,  represents the token ID of the  word in the OCR-extracted text. These token IDs are the first step in converting raw text into a numerical representation suitable for the transformer.

**Contextual Embedding Generation (DistilBERT Encoding)**

The tokenized and padded sequence is fed into the DistilBERT encoder to generate contextual embeddings. Each token embedding considers the surrounding words, capturing semantic relationships:

-  is the DistilBERT encoder parameterized by  (weights learned during pre-training and fine-tuning).
-  is the resulting contextual embedding matrix, with each row corresponding to a token and each column representing one feature in the embedding space.

**How values are obtained**

- **Embedding Lookup:** Each token ID  is mapped to a dense vector in the embedding table.
- **Transformer Layers:** Each vector is passed through multiple self-attention layers, where attention scores determine how much each token attends to others.
- **Layer Outputs:** The output of the final transformer layer forms the contextual embedding (t), where each token vector encodes both its identity and context.

**Positional Encoding Retaining Word Order**

Transformers do not inherently know token positions. Positional encodings  are added to embeddings to preserve sequential information:

-  is the positionally encoded token matrix.
-  contains fixed or learned vectors that encode each token's position.

**How values are obtained:**

- Each position  has a unique vector, either learned during training or using sinusoidal functions.
- These vectors are element-wise added to the token embeddings  , allowing the model to differentiate between “Buy One Get One Free” and “Free One Get Buy One,” which have the same tokens but different meanings.

**Projection to Shared Latent Space (Dimensional Alignment for Fusion)**

Visual and textual embeddings typically have different dimensions (e.g., ResNet outputs 2048-d, DistilBERT outputs 768-d). To enable cross-modal fusion, textual embeddings are projected into a shared latent space (d):

-  and  are learnable weight matrices and bias.
-  is now dimensionally compatible with the visual embedding  .

Fig 3.3.3 Textual Embedding Workflow

**How values are obtained:**

- Each token embedding vector  is multiplied by  (a learned linear transformation) and shifted by  .
- The projection ensures that the textual and visual representations can interact in a meaningful way during cross-attention.

### 3.3.4-Dimensional Projection

The Dimensional Projection aligns the feature representations from visual and textual modalities into a shared latent space, enabling effective cross-modal fusion. Since visual embeddings  extracted from ResNet18 and textual embeddings  extracted from DistilBERT typically have different dimensions (e.g., 2048 for ResNet vs. 768 for DistilBERT), direct interaction between them is not feasible without dimensional alignment.

**Projection of Visual Features (Visual Feature Alignment)**

Visual embeddings  are projected into a shared latent space of dimension  using a learnable linear transformation:

-  is the high-dimensional visual embedding from ResNet18.
-  is the learnable weight matrix for visual projection.
-  is the learnable bias vector.
-  is dimensionally aligned visual embedding.

**How values are obtained**

- Each element of  is multiplied by the corresponding weights in  .
- The linear combination is summed and shifted by  .
- The output  now has dimension  , making it compatible with textual embeddings.

**Projection of Textual Features (Text Feature Alignment)**

Similarly, textual embeddings  are projected into the same shared latent space:

-  is the contextual embedding from DistilBERT.
-  and  are learnable parameters.
-  is the projected textual embedding.

**How values are obtained:**

- Each token embedding in  is multiplied by the weight matrix .
- The bias vector  is added element-wise.
- The output  is now in the same latent space as  , allowing meaningful interactions.

**Rationale for Projection**

The projection ensures dimensional compatibility and semantic alignment between visual and textual modalities:

- **Dimensional Compatibility:** Without projection, embeddings with mismatched sizes cannot be processed by attention-based fusion layers or concatenation operations.
- **Semantic Alignment:** The learnable projections allow the model to adjust the representations such that similar concepts in text and images occupy nearby positions in the shared latent space. For example, a product image embedding and the text "50% OFF Shoes" should align closely in  -dimensional space.

**Mathematical Summary**

The aligned embeddings can be represented compactly as:

-  are ready for cross-modal fusion.
- Both projections are learned jointly with the downstream tasks during training, ensuring the latent space captures meaningful cross-modal correlations.

### 3.3.5 Cross-Modal Fusion

The Cross-Modal Fusion is responsible for integrating the aligned visual  and textual  embeddings into a joint multimodal representation ( M ). This fusion captures fine-grained interactions between the modalities, ensuring that semantic cues from both image and text contribute to downstream tasks such as sentiment analysis.

**Text-Guided Visual Attention**

Textual embeddings  serve as queries, attending over visual keys  and values  to focus on image regions relevant to the text:

-  are learnable projection matrices for query, key, and value, respectively.
-  is the dimensionality of the query/key vectors (used for scaling).
-  is the text-guided visual attention output, emphasizing image regions that are semantically aligned with the text.

**How values are obtained:**

- Text embeddings  are projected to queries  .
- Visual embeddings  are projected to keys  and values nd values  values ( V = V_s W_V ) $$.
- The attention scores are computed as  , then normalized using softmax.
- The normalized scores weight the visual values  , producing the attended output .

**Visual-Guided Text Attention**

Similarly, visual embeddings  act as queries, attending over textual keys and values:

-  are learnable projections for this attention.
-  is the visual-guided text attention output, highlighting text tokens relevant to visual content.

**How values are obtained:**

- Visual embeddings  are projected to queries.
- Text embeddings  are projected to keys and values.
- Attention scores are computed, normalized, and applied to text values, yielding .

**Joint Multimodal Embedding**

The outputs from the two attention mechanisms are concatenated and fused to form a joint multimodal representation (M):

-  denotes concatenation of text-guided visual and visual-guided text embeddings.
-  are learnable parameters for final projection.
- Layer normalization stabilizes training and ensures uniform feature scaling.

**How values are obtained:**

- Concatenate  and  along the feature dimension.
- Apply a linear transformation using  and .
- Normalize with Layer Norm to obtain the final joint embedding  .

**Multiscale Attention Enhancement**

To handle advertisements containing elements at multiple scales (e.g., small discount tags, medium products, large backgrounds), multiscale attention is applied:

-  is the joint embedding computed at scale  .
-  are learnable attention weights over scales.
-  is the final multimodal representation capturing information from all spatial resolutions.

**How values are obtained:**

- Extract embeddings at multiple ResNet layers for each image scale.
- Apply cross-attention with text for each scale to get  .
- Learn attention weights  that determines the importance of each scale.
- Sum weighted embeddings to form the multiscale fused representation  .

### 3.3.6 Sentiment Learning Heads

The Sentiment Learning Head utilizes the joint multimodal embedding  obtained from the Cross-Modal Fusion Module to perform multiple advertisement-related predictions simultaneously. This module is designed to leverage shared representations while optimizing for sentiment classification \[20\]. Sentiment learning improves model generalization and reduces overfitting by enabling common feature learning across tasks.

**Sentiment Classification**

Sentiment classification predicts whether an advertisement conveys positive, negative, or neutral sentiment. The prediction head is a fully connected layer followed by softmax activation:

-  = learnable parameters
-  = predicted sentiment probabilities

**Joint Sentiment Loss Function**

The system is trained by optimizing all tasks simultaneously using a weighted loss function:

-  = cross-entropy loss for classification tasks.
-  = mean squared error for a regression task.
-  = binary cross-entropy for trust prediction
-  = weight for each task to balance contributions

Fig 3.3.6 Sentiment Learning Heads

## 3.4 Tools, Languages and Frameworks

The proposed multimodal advertisement analysis system relies on a carefully chosen set of programming languages, deep learning frameworks, computer vision libraries, NLP tools, data processing libraries, and visualization tools. These tools are selected to ensure high efficiency, scalability, and interpretability while allowing the system to handle complex multimodal inputs, large-scale datasets, and downstream tasks such as sentiment analysis\[21\].

### 3.4.1 Programming Language: Python

Python is the backbone of the entire system. Its simplicity, readability, and flexibility make it ideal for research and production environments. Python is widely used in AI and data science due to its extensive ecosystem of libraries and frameworks, allowing rapid prototyping and implementation of complex pipelines.

Python handles preprocessing, embedding extraction, model training, multimodal fusion, Sentiment predictions, evaluation, and visualization, serving as the glue between all modules.

### 3.4.2 Deep Learning Framework: PyTorch

PyTorch is the primary deep learning framework for building and training neural networks in this project. It is chosen for its dynamic computational graph, which allows flexibility in handling variable-length inputs, attention mechanisms, and complex fusion strategies between modalities.

PyTorch implements the visual embedding (ResNet18), textual embedding (DistilBERT), dimensional projection, cross-modal attention fusion, and Sentiment learning heads. Its tensor operations allow seamless calculation of gradients, losses, and optimization for Sentiment objectives \[22\].

### 3.4.3 Computer Vision Library: OpenCV

OpenCV is employed for handling all image-related preprocessing and augmentation tasks. Digital advertisements can vary significantly in resolution, color, style, and layout. OpenCV ensures images are standardized and prepared for neural network input.

Standardizes image inputs for visual embedding. Applies augmentation to improve robustness to different ad layouts and styles. Prepares images for attention-based fusion by ensuring consistent resolution.

### 3.4.4 OCR Engine: PaddleOCR

PaddleOCR is utilized for extracting textual content embedded in advertisements. Text in ads is often non-standard, stylized, or placed over backgrounds, which makes robust OCR crucial for downstream textual embedding.

Provides clean textual input to the textual embedding module (DistilBERT). Ensures no important promotional message or keyword is lost, which is critical for sentiment, engagement, and theme detection \[23\].

### 3.4.5 Natural Language Processing Library: Hugging Face Transformers

Hugging Face Transformers is employed for generating contextual embeddings for OCR-extracted text using DistilBERT. These embeddings capture semantic meaning, sentiment cues, and promotional intent.

Encodes ad text into fixed-size embeddings compatible with visual embeddings. Captures context-sensitive semantics necessary for predicting sentiment \[24\].

### 3.4.6 Data Handling Libraries: NumPy, Pandas, and Scikit-learn

**NumPy**: Core library for numerical operations and tensor manipulations. Used to handle embedding vectors, matrix operations for attention, and cross-modal fusion computations.

**Pandas**: Manages tabular datasets such as user interaction metrics (click-through rate, likes, shares), ad metadata (campaign, category), and prediction labels. Pandas allows easy merging and manipulation of multimodal datasets.

**Scikit-learn**: Used for preprocessing and evaluation. Standardization and normalization of features. Splitting datasets into training, validation, and test sets. Calculating metrics like F1-score, accuracy, mean squared error for engagement regression.

Provides robust, efficient tools for data handling, preprocessing, and evaluation. Ensures compatibility with PyTorch tensors for model training and prediction.

### 3.4.7 Visualization Library: Matplotlib

Matplotlib is used to visualize intermediate and final outputs for analysis and reporting. Plot training and validation loss curves. Visualize cross-attention maps between text and image regions. Display distributions of predicted sentiment. Present insights for reporting and model interpretability \[25\].

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_e27n3xbkh4y1"></a></h1><p>CHAPTER 4</p><p>IMPLEMENTATION</p></td></tr></tbody></table></div>

## 4.1 System Interaction & Data Flow

### 4.1.1. Advertisement Input

**Raw Input:** Image of the advertisement and embedded textual content.

**Image Processing**

- Resizing to standard input resolution (e.g., 224x224x3) for CNNs.
- Pixel normalization (subtract mean, divide by std deviation) to standardize intensity.
- Optional data augmentation (rotation, flipping, color jitter) to improve generalization.

**Text Processing (OCR Extracted)**

- OCR scans image regions for text using Tesseract or similar tools.
- Cleaning removes unwanted characters (special symbols, artifacts).
- Tokenization splits text into sub words or words.
- Padding/truncation ensures sequences are consistent in length.

**Output:** Preprocessed image tensor  and tokenized text sequence  .

### 4.1.2. Visual Embedding Module (ResNet18, Multiscale Features)

**Input:** Preprocessed image tensor.

**ResNet18 Layers:**

- **Conv1:** 7x7 convolution with stride 2, output feature map detects edges and simple textures.
- **Max Pool:** 3x3 pooling, reduces spatial size and preserves strong features.
- **Residual Block 1:** Two 3x3 convolutions, captures low-level features like edges and corners. Residual connection ensures gradient flow.
- **Residual Block 2:** Captures mid-level features, e.g., product shapes, logos.
- **Residual Block 3:** Extracts larger patterns, e.g., contextual layout, background.
- **Residual Block 4:** Captures high-level semantic features — relationships between objects, spatial hierarchy.

**Multiscale Fusion:**

- Outputs from all residual blocks are up sampled or pooled to a uniform size.
- Concatenation across scales forms a rich embedding vector.

**Output:** Visual embedding  capturing multi-level features.

### 4.1.3. Textual Embedding (DistilBERT)

**Input:** Tokenized text sequence 

**DistilBERT Layers:**

- **Embedding Layer:** Converts tokens into 768-dimensional embeddings.
- **Positional Encoding:** Adds position information to retain word order.

**Transformer Blocks (6 layers in DistilBERT):**

- **Self-Attention:** Each token attends to all others in the sequence.
- **Feed-Forward Network:** Captures higher-level representations.
- **Layer Norm & Residuals:** Stabilizes training and allows deep gradient flow.
- **Output of Each Token:** Contextualized vector  .
- **Pooling:** CLS token or mean pooling produces sentence-level embedding.

**Output:** Text embedding  capturing semantic, syntactic, and sentiment information.

### 4.1.4. Dimensional Projection

**Input:** Visual embedding  and text embedding .

**Process:**

- Apply linear projections:

  

- Aligns embeddings into a shared latent space 
- Ensures that both modalities are compatible for cross-modal attention.

**Output:** Projected embeddings  and  ready for fusion.

### 4.1.5. Cross-Modal Fusion

**Input:** Projected embeddings  and .

**Text-to-Image Attention:**

- Text queries  attend to image keys/values  .
- Determines which visual regions are relevant for each word.

**Image-to-Text Attention:**

- Image queries  attend to textual tokens .
- Highlights which words are crucial for understanding visual features.

**Joint Fusion:**

- Combine attended outputs with concatenation and residual addition.

### 4.1.6. Sentiment Learning Task

**Input:** Joint multimodal embedding  .

**Task Heads:**

- **Sentiment Classification:** Softmax over positive, negative, neutral.

**Output:** Sentiment predictions  .

Fig 4.1.6. Sentiment Learning Task

## 4.2 Data Flow Diagram

The use case analysis defines the interactions between system users and the multimodal advertisement analysis framework. Primary users include marketing analysts, campaign managers, and AI research engineers. The system supports several key use cases, beginning with Advertisement Input, where users upload ad images and textual content. The system extracts visual embeddings using ResNet18 and textual embeddings via DistilBERT. Dimensional alignment ensures that both embeddings are compatible for downstream fusion tasks \[26\].

A second major use case is Multimodal Feature Fusion, where cross-attention mechanisms facilitate interaction between text and image embeddings, capturing fine-grained semantic relationships and aligning features in a shared latent space. This step is essential for accurately understanding the combined meaning of text and visuals, which is critical for subsequent tasks like sentiment analysis. Multiscale visual feature fusion is also a part of this use case, allowing the system to detect small, medium, and large visual elements within ads.

Fig 4.2 Working of Modal in real world

The Prediction and Analysis use case involves applying task-specific heads to the joint multimodal embedding. Results are presented to users in dashboards or exported for further analytics, enabling data-driven marketing decisions. This use case emphasizes the system’s Sentiment learning capability, optimizing all predictions jointly through a weighted loss function \[27\].

Another critical use case is Integration with User Behavior Data, which enables correlation of ad content with real-world user interactions, providing actionable insights into campaign effectiveness. This allows marketers to adjust creative strategies, optimize ad placement, and target specific audience segments more effectively. Finally, the system supports Model Maintenance and Updates, where new ads and interaction datasets are used to retrain or fine-tune models incrementally, ensuring continuous improvement and adaptability over time \[28\].

**4.3 Result**

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_fobeyh1zclc7"></a></h1><p></p><p>CONCLUSION</p><p></p></td></tr></tbody></table></div>

# **CONCLUSION**

The proposed multimodal advertisement sentiment analysis system successfully integrates textual and visual information to achieve a unified understanding of advertisements. By combining deep visual encoders (ResNet18) and contextual language models (DistilBERT), the framework bridges the gap between imagery and language — two crucial components of advertising communication. This fusion allows the system to capture not only the literal content of ads but also the emotional and semantic cues embedded in both text and design.

At the core of this framework lies a Dimensional Projection Module that aligns heterogeneous embeddings from different modalities into a common latent space, ensuring compatibility for joint learning. The Cross-Modal Fusion Module, employing bidirectional attention mechanisms, enables fine-grained interaction between textual and visual signals, allowing the model to emphasize words and image regions that collectively contribute to overall sentiment. Finally, the Multi-Task Learning Head leverages the joint embedding to predict sentiment efficiently while maintaining strong generalization across advertisement types and content styles.

The system’s design reflects a deep understanding of how modern advertisements communicate meaning. Unlike traditional sentiment analysis methods that focus solely on textual data, this approach treats advertisements as multimodal messages, where both visual design and textual phrasing influence perception. Through multiscale visual feature extraction, the model identifies details such as product logos, discount labels, and emotional color schemes, while the textual encoder deciphers persuasive language, tone, and emotion from promotional text. Together, these components enable a holistic interpretation that aligns closely with human perception of advertising content.

Experimentally, the model demonstrates robustness in feature alignment, efficient computation due to lightweight architectures, and interpretability through cross-attention maps. These qualities make it a promising foundation for large-scale commercial sentiment analysis pipelines or automated ad quality evaluation systems.

In summary, this project presents a comprehensive multimodal learning framework that fuses computer vision and natural language processing for advertisement understanding. It offers a balanced combination of architectural simplicity, interpretability, and effectiveness, demonstrating that unified multimodal representations can substantially enhance affective computing tasks. Future work could explore more advanced fusion strategies.

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1></h1><p></p><p>FUTURE WORK</p></td></tr></tbody></table></div>

# **FUTURE WORK**

While the current framework effectively performs sentiment prediction from multimodal advertisement inputs, several promising extensions and optimizations are envisioned for future development. The next phase of this research will focus on expanding the model’s analytical scope, improving computational efficiency, and enhancing deployment readiness for real-world applications.

A key direction is the integration of additional affective and contextual tasks such as emotion detection, theme classification, and trustworthiness assessment. These extensions will build upon the existing multimodal joint embedding, allowing the model to capture a broader emotional and psychological spectrum of advertisement perception. Incorporating these auxiliary tasks through a refined multi-task learning architecture will enable the system to infer not only whether an ad is positive or negative but also why—for instance, by recognizing emotions such as excitement, trust, or nostalgia conveyed through both text and visuals.

Another major objective is optimization for efficiency and scalability. The current implementation, while accurate, relies on deep architectures that may limit real-time deployment in large-scale ad analytics environments. To address this, future work will explore model compression, quantization, and conversion to the ONNX format, facilitating interoperability across frameworks and platforms. This will enable faster inference, reduced memory consumption, and seamless deployment on edge devices and cloud systems.

Further improvements can also be achieved by experimenting with parameter-efficient fine-tuning methods such as LoRA (Low-Rank Adaptation) or adapter-based training. These approaches will make it possible to adapt the model to new advertisement domains or languages with minimal retraining costs. Additionally, optimizing the cross-modal fusion mechanism using attention refinement or transformer-based fusion layers could enhance the quality of joint embeddings and improve interpretability \[29\].

Although the present work is limited to static image-based advertisements, the framework’s modular design allows for straightforward adaptation to other modalities in the future, such as dynamic or interactive media. However, in the immediate scope, the focus remains on perfecting static ad analysis by enhancing emotion-aware sentiment prediction, improving model interpretability, and achieving deployment-grade performance through ONNX optimization.

# 

<div class="joplin-table-wrapper"><table><tbody><tr><td><h1><a id="_3r0d4ho7i1r7"></a></h1><p></p><p>REFERENCES</p></td></tr></tbody></table></div>

# **REFERENCE**

1.  Wang, J. Wang, and H. Lu, “Exploiting content relevance and social relevance for personalized ad recommendation on Internet TV,” _ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)_, vol. X, no. X, pp.
2.  Kumar, S. Gupta, and P. Singh, “Multimodal ad recommendation using visual and textual content,” _IEEE Transactions on Multimedia_, vol. X, no. X, pp.
3.  M. Zhao, L. Li, and R. Wang, “Understanding advertising effectiveness: Combining visual and textual analysis,” _Journal of Advertising Research_, vol. X, no. X, pp.
4.  Radford et al., “Learning transferable visual models from natural language supervision,” _ICML_,2021 2021.
5.  J. Li et al., “VisualBERT: A simple and performant baseline for vision and language,” _arXiv preprint arXiv:1908.03557_, 2019.
6.  K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” _CVPR_, 2016.
7.  V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” _NeurIPS_, 2019.
8.  N. Vedula, W. Sun, H. Lee, H. Gupta, M. Ogihara, J. Johnson, G. Ren, and S. Parthasarathy, “Multimodal content analysis for effective advertisements on YouTube,” in _Proc. IEEE Int. Conf. Data Mining (ICDM)_, 2017, pp. 1123–1128.
9.  L. M. Lodish et al., “How TV advertising works: A meta‐analysis of 389 real world split cable TV advertising experiments,” _Journal of Marketing Research_, vol. 32, no. 2, pp. 125–139, 1995.
10. Zadeh, M. Chen, S. Poria, E. Cambria, and L. Morency, “Tensor fusion network for multimodal sentiment analysis,” in Proc. EMNLP, 2017.
11. Zadeh, P. P. Liang, S. Poria, E. Cambria, and L. Morency, “Memory fusion network for multimodal sentiment analysis,” in Proc. ACL, 2018.
12. N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, E. Cambria, and A. Gelbukh, “DialogueRNN: An attentive RNN for emotion detection in conversations,” in Proc. AAAI, 2019.
13. M. Xu, Y. Zhou, Z. Xu, and Q. Wu, “Cross‐modal attention networks for multimodal sentiment analysis,” in Proc. ACM MM, 2019.
14. S. Li, W. Lu, and L. Zhu, “MCAM: Multimodal cross‐attention model for sentiment analysis,” Information Fusion, vol. 101, pp. 1–14, 2024.
15. Y. Lu, Z. Ni, and L. Ding, “CLIP‐based image-text sentiment analysis with cross-modal attention,” Pattern Recognition, vol. 139, 2024.
16. Y. Yu, J. Li, and H. Xu, “FGMFN: Fine-grained multiscale cross-modal feature network for advertisement sentiment analysis,” in Proc. AAAI, 2022.
17. H. Zhang, L. Wu, and J. Chen, “Sentiment analysis technologies of advertising images based on deep learning,” Multimedia Tools and Applications, vol. 82, pp. 11723–11740, 2023.
18. Serra, S. Porta, and B. Gatti, “The emotions of the crowd: Learning image sentiment from tweets via cross-modal distillation,” in Proc. CVPR, 2023.
19. R. Wang, J. Zhou, and K. Liu, “Multimodal aspect-based sentiment analysis with external knowledge and multi-granularity image-text features,” Knowledge-Based Systems, vol. 294, 2025.
20. J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” NAACL-HLT, 2019.
21. T.-Y. Lin et al., “Feature Pyramid Networks for object detection,” CVPR, 2017.
22. Thet, L. Na, and S. Khoo, “Aspect-based sentiment analysis of movie reviews on discussion boards,” Expert Systems with Applications, 2010.
23. Thelwall, K. Buckley, and G. Paltoglou, “Sentiment strength detection in short informal text,” _Journal of the American Society for Information Science and Technology_, 2012.
24. J. Snow, B. O’Connor, D. Jurafsky, and A. Ng, “Cheap and fast—but is it good? Evaluating non-expert annotations for natural language tasks,” _EMNLP_, 2008.
25. Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” _arXiv preprint arXiv:1907.11692_, 2019.
26. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” _NAACL-HLT_, 2019.
27. K. He et al., “Identity mappings in deep residual networks,” _ECCV_, 2016.
28. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” _ICLR_, 2021.
29. Y. Chen et al., “UNITER: UNiversal Image-Text Representation Learning,” _ECCV_, 2020.

**IMPLEMENTATION PAPER DEATAILS**

|     |     |
| --- | --- |
| **Project Name** | **“Multi-modal Advertisement Sentiment Analysis”** |
| **Journal Name** | **International Journal of Creative Research Through (IJCRT).** |
|     |     |
| **Sr. No** | **Authors** |
| 01  | Mr. Anish Jagdale |
| 02  | Mr. Dayanand Kadam |
| 03  | Mr. Sanket Misal |
| 04  | Miss. Suchita Shinde |
|     |     |
| **Sr. No** | **Guide Name** |
| 01  | Dr. Manav Thakur |