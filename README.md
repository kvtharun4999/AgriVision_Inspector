# AgriVision Turmeric Inspector 🌿
### *Intelligent Disease Diagnosis, Foliar Severity Quantification & TNAU Bilingual Advisory System*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://tensorflow.org/)
[![Architecture](https://img.shields.io/badge/Backbone-MobileNetV2-brightgreen.svg)](https://arxiv.org/abs/1801.04381)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.43%25-success.svg)](#7-model-evaluation-confusion-matrix--performance-metrics-9843)
[![Dataset](https://img.shields.io/badge/Dataset%20Size-9%2C771%20Images-blueviolet.svg)](#6-multi-source-dataset-harmonization-9771-unique-images)
[![TFLite](https://img.shields.io/badge/Model%20Size-2.72%20MB%20(Quantized)-yellow.svg)](#56-model-quantization--tflite-export)
[![Framework](https://img.shields.io/badge/Backend-Flask%20WSGI-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Region](https://img.shields.io/badge/Target%20Region-Erode%20%26%20Perundurai%2C%20TN-important.svg)](#1-executive-summary--regional-agronomic-context)
[![Language](https://img.shields.io/badge/Bilingual-English%20%7C%20%E0%AE%A4%E0%AE%AE%E0%AE%BF%E0%AE%B4%E0%AF%8D-teal.svg)](#9-agronomic-compendium-pathogens-symptoms-treatments--tnau-protocols)

---

## 📑 Comprehensive Table of Contents

1. [Executive Summary & Regional Agronomic Context](#1-executive-summary--regional-agronomic-context)
2. [Key Capabilities & Technical Innovations](#2-key-capabilities--technical-innovations)
3. [End-to-End System Architecture & Data Flow](#3-end-to-end-system-architecture--data-flow)
4. [The Strict 3-Tier Gatekeeper Validation System](#4-the-strict-3-tier-gatekeeper-validation-system)
   - [4.1 Tier 1: File Integrity & Memory Buffer Decoding](#41-tier-1-file-integrity--memory-buffer-decoding)
   - [4.2 Tier 2: OpenCV HSV Foliage Color Space Segmentation](#42-tier-2-opencv-hsv-foliage-color-space-segmentation)
   - [4.3 Tier 3: Deep Model Classification & Confidence Threshold Gating](#43-tier-3-deep-model-classification--confidence-threshold-gating)
5. [Machine Learning Deep Dive: MobileNetV2 Architecture & Transfer Learning](#5-machine-learning-deep-dive-mobilenetv2-architecture--transfer-learning)
   - [5.1 Backbone Architecture: Inverted Residuals & Linear Bottlenecks](#51-backbone-architecture-inverted-residuals--linear-bottlenecks)
   - [5.2 Real-Time Data Augmentation Pipeline](#52-real-time-data-augmentation-pipeline)
   - [5.3 Custom Classification Head Architecture & Parameters Table](#53-custom-classification-head-architecture--parameters-table)
   - [5.4 Two-Phase Training Strategy: Frozen Backbone vs. Fine-Tuning](#54-two-phase-training-strategy-frozen-backbone-vs-fine-tuning)
   - [5.5 Balanced Class Weights Formulation](#55-balanced-class-weights-formulation)
   - [5.6 Model Quantization & TFLite Export](#56-model-quantization--tflite-export)
6. [Multi-Source Dataset Harmonization (9,771 Unique Images)](#6-multi-source-dataset-harmonization-9771-unique-images)
   - [6.1 Composition of Integrated Datasets (Datasets A, B, and C)](#61-composition-of-integrated-datasets-datasets-a-b-and-c)
   - [6.2 MD5 Hash-Based Deduplication Algorithm](#62-md5-hash-based-deduplication-algorithm)
   - [6.3 Consolidated Class Distribution & Stratified Splitting](#63-consolidated-class-distribution--stratified-splitting)
7. [Model Evaluation, Confusion Matrix & Performance Metrics (98.43%)](#7-model-evaluation-confusion-matrix--performance-metrics-9843)
   - [7.1 Confusion Matrix Analysis](#71-confusion-matrix-analysis)
   - [7.2 Mathematical Definitions of Performance Metrics](#72-mathematical-definitions-of-performance-metrics)
   - [7.3 Per-Class Precision, Recall, Specificity, and F1-Scores](#73-per-class-precision-recall-specificity-and-f1-scores)
8. [Computer Vision Disease Severity Quantification (OpenCV HSV)](#8-computer-vision-disease-severity-quantification-opencv-hsv)
   - [8.1 HSV Color Space Conversion & Mathematical Formulation](#81-hsv-color-space-conversion--mathematical-formulation)
   - [8.2 Dual-Mask Necrotic Spot Segmentation](#82-dual-mask-necrotic-spot-segmentation)
   - [8.3 Morphological Closing & Contour Operations](#83-morphological-closing--contour-operations)
   - [8.4 Clinical Severity Categorization Scale](#84-clinical-severity-categorization-scale)
9. [Agronomic Compendium: Pathogens, Symptoms, Treatments & TNAU Protocols](#9-agronomic-compendium-pathogens-symptoms-treatments--tnau-protocols)
   - [9.1 Leaf Spot (*Colletotrichum capsici*) / இலைப்புள்ளி நோய்](#91-leaf-spot-colletotrichum-capsici--இலைப்புள்ளி-நோய்)
   - [9.2 Leaf Blotch (*Taphrina maculans*) / இலைக்கருகல் நோய்](#92-leaf-blotch-taphrina-maculans--இலைக்கருகல்-நோய்)
   - [9.3 Aphids Infestation (*Aphis gossypii*) / அசுவினி நோய்](#93-aphids-infestation-aphis-gossypii--அசுவினி-நோய்)
   - [9.4 Dry Leaf / காய்ந்த இலை](#94-dry-leaf--காய்ந்த-இலை)
   - [9.5 Rhizome Rot (*Pythium aphanidermatum*) / கிழங்கு அழுகல் நோய்](#95-rhizome-rot-pythium-aphanidermatum--கிழங்கு-அழுகல்-நோய்)
   - [9.6 Healthy Turmeric Foliage / ஆரோக்கியமான இலை](#96-healthy-turmeric-foliage--ஆரோக்கியமான-இலை)
10. [Dual-Engine Agricultural Advisory & Sub-Millisecond Caching](#10-dual-engine-agricultural-advisory--sub-millisecond-caching)
    - [10.1 Structured Anti-Hallucination System Prompting](#101-structured-anti-hallucination-system-prompting)
    - [10.2 Multi-Provider Support (Google Gemini & OpenAI)](#102-multi-provider-support-google-gemini--openai)
    - [10.3 High-Speed Local Caching Architecture (`info_cache.json`)](#103-high-speed-local-caching-architecture-info_cachejson)
    - [10.4 Zero-Downtime Offline Fallback Database (`diseases.json`)](#104-zero-downtime-offline-fallback-database-diseasesjson)
11. [Frontend User Experience & Interactive Design (Google Stitch)](#11-frontend-user-experience--interactive-design-google-stitch)
    - [11.1 Agricultural Design System & Palette Tokens](#111-agricultural-design-system--palette-tokens)
    - [11.2 Native Camera & Drag-and-Drop Dropzone](#112-native-camera--drag-and-drop-dropzone)
    - [11.3 Animated Cybernetic Laser Scanline](#113-animated-cybernetic-laser-scanline)
    - [11.4 Dynamic SVG Circular Progress Ring](#114-dynamic-svg-circular-progress-ring)
    - [11.5 Actionable Accordions & Treatment Steps](#115-actionable-accordions--treatment-steps)
    - [11.6 Instant Client-Side Bilingual Translation Engine](#116-instant-client-side-bilingual-translation-engine)
    - [11.7 Printable Diagnostic Dossier & PDF Generation (`/report/pdf`)](#117-printable-diagnostic-dossier--pdf-generation-reportpdf)
    - [11.8 LocalStorage Scan History Log (`/history`)](#118-localstorage-scan-history-log-history)
12. [Comprehensive Codebase & File-by-File Reference](#12-comprehensive-codebase--file-by-file-reference)
13. [Installation, Dependencies & Step-by-Step Execution Guide](#13-installation-dependencies--step-by-step-execution-guide)
14. [Command-Line Interface (CLI) Training Guide](#14-command-line-interface-cli-training-guide)
15. [REST API Specification (Endpoints, Payloads, Responses)](#15-rest-api-specification-endpoints-payloads-responses)
16. [Production Deployment & Edge Device Guide (Gunicorn, Nginx, TFLite on Raspberry Pi)](#16-production-deployment--edge-device-guide-gunicorn-nginx-tflite-on-raspberry-pi)
17. [Troubleshooting & Frequently Asked Questions (FAQ)](#17-troubleshooting--frequently-asked-questions-faq)
18. [Academic Citations, References & Official Disclaimer](#18-academic-citations-references--official-disclaimer)

---

## 1. Executive Summary & Regional Agronomic Context

### 1.1 The Erode & Perundurai Turmeric Belt
**Erode**, located in western Tamil Nadu, India, is celebrated as the **"Yellow City" (மஞ்சள் மாநகரம்)**. The district accounts for a massive share of the state's turmeric production, supported by the Bhavani and Amaravathi river basins, fertile clay loam soils, and ideal tropical temperatures ($20^\circ\text{C}$ to $35^\circ\text{C}$). Surrounding agrarian hubs such as **Perundurai**, Modakkurichi, Kodumudi, and Bhavani cultivate thousands of hectares of *Curcuma longa* (primarily local cultivars such as *Erode Local*, *BSR-1*, *BSR-2*, and *PTS-10*).

### 1.2 The Agricultural Problem Statement
Turmeric is a long-duration commercial spice crop spanning **8 to 9 months** (planting in May–July, harvesting in January–March). During the southwest and northeast monsoon seasons (July through November), the crop faces extreme pathological pressures:
1. **Foliar Blights**: High relative humidity ($>80\%$) and continuous overcast days trigger explosive airborne outbreaks of **Leaf Blotch (*Taphrina maculans*)** and rain-splash outbreaks of **Leaf Spot (*Colletotrichum capsici*)**.
2. **Photosynthetic Impairment & Yield Loss**: Foliar pathogens destroy chloroplast-rich laminar tissue. When leaf necrotic damage exceeds $25\%$, the translocation of photosynthates to subterranean rhizomes plummets, causing **15% to 30% loss in finger rhizome weight** and poor curcumin content.
3. **Pest Misdiagnosis**: Early sap-sucking pest infestations by **Aphids (*Aphis gossypii*)** produce chlorotic mottling and honeydew sooty mold that farmers often mistake for fungal blights, resulting in unnecessary, costly, and ineffective fungicide sprays.
4. **The "Garbage-In, Garbage-Out" Flaw in Generic AI Tools**: Generic agricultural apps will eagerly classify photographs of shoes, hands, weeds, or tractor tires as having "Leaf Spot" or "Blotch", causing distrust among rural farmers.
5. **Language & Accessibility Barrier**: Most machine learning prototypes operate in English. Rural farmers require actionable Tamil guidance with authentic regional vernacular (e.g., விரலி மஞ்சள், இலைப்புள்ளி, மான்கோசெப் தெளிப்பு).

### 1.3 The AgriVision Solution
AgriVision Turmeric Inspector provides an end-to-end academic decision-support platform designed specifically for this region:
- **Strict 3-Tier Pre-Classification Validation**: Completely rejects non-turmeric uploads and non-leaf objects before disease classification begins.
- **High-Accuracy MobileNetV2 Deep Learning**: Trained on **9,771 images** across 7 classes with **98.43% test accuracy**.
- **Quantified Foliar Severity Measurement**: Measures the exact percentage of leaf tissue affected using OpenCV HSV segmentation.
- **Authoritative TNAU & ICAR Knowledge Base**: Provides official fungicide dosages and preventive cultural practices.
- **Instant Bilingual Switching**: Seamless toggling between English and தமிழ்.
- **Edge-Ready TFLite Export**: 2.72 MB quantized model engineered for offline deployment on smartphones and Raspberry Pi devices.

---

## 2. Key Capabilities & Technical Innovations

| Capability | Implementation Detail | Technical Rationale |
|:---|:---|:---|
| **Gatekeeper Verification** | 3-stage validation in `app.py`: File $\to$ Foliage HSV $\to$ ML Confidence | Prevents hallucinated disease diagnosis on non-leaf images |
| **Transfer Learning** | MobileNetV2 pretrained on ImageNet with custom head | Lightweight (1.28M base params), ultra-fast on standard CPU |
| **Two-Phase Training** | 8 epochs frozen base + 4 epochs fine-tuning top 30 layers | Maximizes feature retention while adapting to leaf textures |
| **Multi-Source Deduplication** | MD5 hash-level image comparison across 3 distinct datasets | Prevents identical images from leaking across train/test splits |
| **Severity Segmentation** | HSV green vs. necrotic spot color masking + morphological closing | Provides objective, measurable disease damage percentage |
| **Dual Advisory Engine** | Gemini 1.5 Flash / OpenAI GPT-4o-mini + local `diseases.json` | Live AI insights with zero downtime during internet outages |
| **Sub-Millisecond Cache** | Persistent local key-value store in `info_cache.json` | 0 ms response on repeated queries; preserves API quota |
| **Clinical PDF Dossier** | Dynamic PDF generation via ReportLab (`/report/pdf`) | Farmers can print or share diagnostic reports via WhatsApp |

---

## 3. End-to-End System Architecture & Data Flow

```text
                                 ╔═══════════════════════════╗
                                 ║   Farmer Uploads Image    ║
                                 ║  (Web Dropzone / Camera)  ║
                                 ╚═════════════╦═════════════╝
                                               ║
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │  TIER 1: FILE INTEGRITY CHECK             │
                         │  • Extension: .jpg, .jpeg, .png, .webp    │
                         │  • Payload limit: <= 5 MB (5,242,880 B)   │
                         │  • In-memory decoding: cv2.imdecode       │
                         └─────────────────────┬─────────────────────┘
                                               │ [Pass]
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │  TIER 2: PLANT FOLIAGE CHECK (OpenCV)     │
                         │  • Convert BGR to HSV color space         │
                         │  • Foliage Mask: Green (15-95) | Tan (5-30│
                         │  • Condition: Foliage Area Ratio >= 5.0%  │
                         └─────────────────────┬─────────────────────┘
                                               │ [Pass]
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │  TIER 3: MODEL INFERENCE & CONFIDENCE GATE│
                         │  • Resize to 224x224 RGB, preprocess [-1,1│
                         │  • MobileNetV2 Softmax Prediction (7 cls) │
                         │  • Condition: Top Class Confidence >= 70% │
                         │  • Condition: Top Class is Foliar Turmeric│
                         └─────────────────────┬─────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               │                                                               │
        [Fails Any Check]                                             [All Checks Passed]
               ▼                                                               ▼
╔═════════════════════════════╗                               ╔═════════════════════════════╗
║  ❌ REJECTION SCREEN        ║                               ║  ✅ CROP CONFIRMED          ║
║  invalid_warning.html       ║                               ║  "Turmeric leaf confirmed"  ║
║  "Please upload a correct   ║                               ╚═══════════════╦═════════════╝
║   turmeric leaf photo."     ║                                               ║
╚═════════════════════════════╝                                               ▼
                                                              ┌─────────────────────────────┐
                                                              │ OPENCV SEVERITY SEGMENTATION│
                                                              │ • Healthy Leaf Mask (H20-90)│
                                                              │ • Necrotic Spot Mask        │
                                                              │ • Infected Area % =         │
                                                              │   (Spots / Total Leaf) * 100│
                                                              │ • Severity: Mild/Mod/Severe │
                                                              └───────────────┬─────────────┘
                                                                              ║
                                                                              ▼
                                                              ┌─────────────────────────────┐
                                                              │ AGRONOMIC ADVISORY ENGINE   │
                                                              │ 1. Check info_cache.json    │
                                                              │ 2. Query Gemini/OpenAI API  │
                                                              │ 3. Fallback: diseases.json  │
                                                              └───────────────┬─────────────┘
                                                                              ║
                                                                              ▼
                                                              ╔═════════════════════════════╗
                                                              ║  BILINGUAL DASHBOARD (UI)   ║
                                                              ║  result.html / JSON API     ║
                                                              ║  • SVG Animated Gauge Ring  ║
                                                              ║  • Numbered TNAU Solutions  ║
                                                              ║  • English / தமிழ் Switcher ║
                                                              ║  • ReportLab PDF Download   ║
                                                              ╚═════════════════════════════╝
```

---

## 4. The Strict 3-Tier Gatekeeper Validation System

Field testing reveals that farmers often test computer vision systems with blurry images, background objects, cattle, or weeds. Without pre-validation, a convolutional neural network will output an arbitrary classification with artificial confidence. AgriVision solves this with a **sequential 3-tier gatekeeper pipeline** in `app.py`.

```text
+--------------------------------------------------------------------------------------------------+
|                                    3-TIER GATEKEEPER VALIDATION                                  |
+---------------------+---------------------------------------+------------------------------------+
| Validation Tier     | Trigger Criteria                      | Rejection Notice                   |
+---------------------+---------------------------------------+------------------------------------+
| Tier 1: File Check  | Corrupt image, invalid MIME, > 5MB    | "File format not supported /       |
|                     |                                       |  File exceeds 5 MB limit"          |
+---------------------+---------------------------------------+------------------------------------+
| Tier 2: Foliage HSV | Non-plant objects, soil, hands,       | "Insufficient leaf foliage pixels  |
|                     | background walls (Foliage Ratio < 5%) |  detected. Upload a clear leaf."   |
+---------------------+---------------------------------------+------------------------------------+
| Tier 3: Model Gate  | Model Confidence < 70.0% OR           | "Low match confidence / Identified |
|                     | Class is "Not turmeric leaf" / Root   |  as non-turmeric material"         |
+---------------------+---------------------------------------+------------------------------------+
```

### 4.1 Tier 1: File Integrity & Memory Buffer Decoding
1. **Extension Filtering**: Only `.jpg`, `.jpeg`, `.png`, and `.webp` are permitted.
2. **Byte Buffer Check**: In-flight upload size is capped at $5 \times 1024 \times 1024 = 5,242,880\text{ bytes}$ (5 MB) via `app.config["MAX_CONTENT_LENGTH"]`.
3. **Safe Memory Decode**:
   ```python
   np_arr = np.frombuffer(file_bytes, np.uint8)
   img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
   if img_bgr is None or img_bgr.size == 0:
       raise ValueError("Corrupt image data")
   ```
   This prevents arbitrary code execution and catches malformed image payloads before filesystem storage.

### 4.2 Tier 2: OpenCV HSV Foliage Color Space Segmentation
RGB representations fluctuate heavily under natural sunlight, cloud shadows, and camera white balance. AgriVision converts the image to the **Hue-Saturation-Value (HSV)** cylinder:
* **Green / Yellow Foliage Mask ($M_{\text{green}}$)**:
  $$\text{Hue} \in [15, 95], \quad \text{Saturation} \in [20, 255], \quad \text{Value} \in [20, 255]$$
* **Necrotic Chlorotic Foliage Mask ($M_{\text{brown}}$)**:
  $$\text{Hue} \in [5, 30], \quad \text{Saturation} \in [25, 255], \quad \text{Value} \in [20, 240]$$
* **Combined Plant Mask**:
  $$M_{\text{plant}} = M_{\text{green}} \lor M_{\text{brown}}$$
* **Foliage Ratio Calculation**:
  $$\text{Ratio} = \frac{\sum M_{\text{plant}}}{\text{Height} \times \text{Width}}$$
* **Gate Decision**: If $\text{Ratio} < 0.05$ (less than 5% of the total pixel area matches plant tissue), the upload is immediately rejected with:
  > *"Insufficient leaf foliage pixels detected. Please photograph a green, yellow, or brown turmeric leaf."*

### 4.3 Tier 3: Deep Model Classification & Confidence Threshold Gating
If the image passes Tiers 1 and 2, it is passed through MobileNetV2:
1. **Confidence Thresholding**:
   $$C_{\text{top}} = \max_{i} P(y = c_i \mid x) \times 100\%$$
   If $C_{\text{top}} < 70.0\%$, the leaf is rejected for ambiguity:
   > *"Low match confidence ($C_{\text{top}}\% < 70\%$). Please photograph under uniform daylight."*
2. **Non-Foliar / Negative Class Rejection**:
   If the predicted class is `"Not turmeric leaf"` or rhizome roots (`"Rhizome Disease Root"`, `"Rhizome Healthy Root"`), the app prevents misdiagnosis:
   > *"This inspector analyzes foliar leaves. For rhizome rot, examine seed tuber drainage."*

---

## 5. Machine Learning Deep Dive: MobileNetV2 Architecture & Transfer Learning

```text
                           INPUT TENSOR: (224, 224, 3)
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    REAL-TIME DATA AUGMENTATION          │
                  │    • Random Horizontal & Vertical Flip  │
                  │    • Random Rotation (+/- 15%)          │
                  │    • Random Brightness (+/- 20%)        │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    MOBILENETV2 PREPROCESSING            │
                  │    x_norm = (x / 127.5) - 1.0           │
                  │    Maps [0, 255] -> [-1.0, 1.0]         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    MOBILENETV2 BACKBONE (ImageNet)      │
                  │    • 154 layers total                   │
                  │    • Inverted Residual Bottlenecks      │
                  │    • Depthwise Separable Convolutions   │
                  │    Output: (7, 7, 1280) Feature Map     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │    CUSTOM CLASSIFICATION HEAD           │
                  │    • GlobalAveragePooling2D -> (1280)   │
                  │    • Dropout (p = 0.3)                  │
                  │    • Dense (128, activation="relu")     │
                  │    • Dropout (p = 0.2)                  │
                  │    • Dense (7, activation="softmax")    │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                         PREDICTED PROBABILITY VECTOR
                         [p0, p1, p2, p3, p4, p5, p6]
```

### 5.1 Backbone Architecture: Inverted Residuals & Linear Bottlenecks
Standard convolutional layers compute spatial and channel-wise correlations simultaneously, requiring heavy FLOPs. MobileNetV2 splits this using **Depthwise Separable Convolutions**:
1. **$1\times 1$ Expansion Convolution**: Projects low-dimensional representations to higher dimensions.
2. **$3\times 3$ Depthwise Convolution**: Performs lightweight spatial filtering.
3. **$1\times 1$ Projection (Linear Bottleneck)**: Compresses features back to a lower dimension without a non-linear activation (preserving manifold information).
4. **Residual Shortcut Connection**: Directly connects inputs and outputs when input/output channels match.

### 5.2 Real-Time Data Augmentation Pipeline
To prevent overfitting on laboratory backgrounds and simulate harsh sunlight, shadow, and orientation variations in the field, augmentation is applied dynamically on the GPU/CPU:
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=42),
    layers.RandomRotation(0.15, seed=42),
    layers.RandomBrightness(0.2, seed=42),
], name="data_augmentation")
```

### 5.3 Custom Classification Head Architecture & Parameters Table

| Layer Name | Layer Type | Output Dimension | Parameter Count | Activation Function |
|:---|:---|:---:|:---:|:---:|
| `turmeric_leaf_input` | InputLayer | `(None, 224, 224, 3)` | 0 | None |
| `data_augmentation` | Sequential | `(None, 224, 224, 3)` | 0 | None |
| `mobilenetv2_1.00_224` | Functional Backbone | `(None, 7, 7, 1280)` | 2,257,984 | ReLU6 |
| `global_avg_pool` | GlobalAveragePooling2D | `(None, 1280)` | 0 | None |
| `head_dropout` | Dropout ($p=0.3$) | `(None, 1280)` | 0 | None |
| `dense_features` | Dense | `(None, 128)` | 163,968 | ReLU |
| `dropout_2` | Dropout ($p=0.2$) | `(None, 128)` | 0 | None |
| `prediction` | Dense (Output) | `(None, 7)` | 903 | Softmax |
| **Total Parameters** | — | — | **2,422,855** | — |
| **Trainable Params (Phase 1)** | — | — | **164,871** | — |
| **Trainable Params (Phase 2)** | — | — | **1,093,895** | — |

### 5.4 Two-Phase Training Strategy: Frozen Backbone vs. Fine-Tuning

#### Phase 1: Classification Head Training
- **Backbone**: `base_model.trainable = False` (all 154 ImageNet layers frozen).
- **Optimizer**: Adam ($\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$).
- **Loss**: Categorical Crossentropy.
- **Epochs**: 8 epochs.
- **Callbacks**: `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)`.

#### Phase 2: Fine-Tuning Top Layers
- **Backbone Unfreezing**:
  ```python
  base_model.trainable = True
  for layer in base_model.layers[:-30]:
      layer.trainable = False
  ```
- **Top 30 layers**: Unfrozen to adapt high-level visual features (vein necrosis, concentric rings, aphid sooty mold) to turmeric plant specifics.
- **Optimizer**: Adam with reduced learning rate ($\eta = 10^{-5}$) to prevent catastrophic forgetting.
- **Epochs**: 4 epochs.
- **Callbacks**: `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`.

### 5.5 Balanced Class Weights Formulation
To compensate for class distribution differences between common classes (Leaf Blotch: 2,377 images) and smaller classes (Dry Leaf: 812 images), training applies inverse-frequency balanced class weights:
$$W_j = \frac{N}{K \times N_j}$$
Where:
- $N$ = Total number of training samples ($6,839$).
- $K$ = Number of target classes ($7$).
- $N_j$ = Number of training samples in class $j$.

### 5.6 Model Quantization & TFLite Export
To enable inference on low-power devices, the trained model is quantized using TensorFlow Lite:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
```
- **FP32 Keras Model**: `23.85 MB`
- **Quantized TFLite Model**: `2.72 MB` (**88.6% size reduction**)
- **Inference Latency**: $\approx 18\text{ ms}$ on standard x86 CPU, $\approx 35\text{ ms}$ on Raspberry Pi 4 ARM Cortex-A72.

---

## 6. Multi-Source Dataset Harmonization (9,771 Unique Images)

AgriVision combines data from three distinct repositories to build a large-scale turmeric disease dataset:

```text
                                  DATASET REPOSITORIES
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│   Dataset A (Local)     │   │   Dataset B (HuggingFace│   │   Dataset C (High-Res)  │
│ Turmeric Plant Disease  │   │ Project-AgML Parquet    │   │ Image Dataset for       │
│ Augmented Dataset       │   │ Streaming Extraction    │   │ Turmeric Leaf Disease   │
│ 4,548 images (5 classes)│   │ 865 images (4 classes)  │   │ 4,361 images (4 classes)│
└────────────┬────────────┘   └────────────┬────────────┘   └────────────┬────────────┘
             │                             │                             │
             └─────────────────────────────┼─────────────────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │  MD5 Hash Deduplication & Name Mapping│
                       │  (Aphids, Spot, Blotch, Healthy, etc.)│
                       └───────────────────┬───────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │  Unified Dataset: 9,771 Unique Images │
                       │  Stratified Split: 70% / 15% / 15%    │
                       └───────────────────┬───────────────────┘
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
           Train Set (70%)          Val Set (15%)            Test Set (15%)
            6,839 images             1,466 images             1,466 images
```

### 6.1 Composition of Integrated Datasets (Datasets A, B, and C)

1. **Dataset A: `Turmeric Plant Disease Augmented Dataset/`**
   - **Origin**: Locally archived agricultural repository.
   - **Classes Included**: `Dry Leaf`, `Healthy Leaf`, `Leaf Blotch`, `Rhizome Disease Root`, `Rhizome Healthy Root`.
   - **Count**: 4,548 unique images.

2. **Dataset B: `dataset_hf/` (`Project-AgML/turmeric_leaf_disease_classification`)**
   - **Origin**: Hugging Face Spices Dataset (streamed & decoded from Parquet).
   - **Classes Included**: `Aphids_Disease`, `Leaf_Blotch`, `Healthy_Leaf`, `Leaf_Spot`.
   - **Count**: 865 images.

3. **Dataset C: `Image Dataset for Turmeric Plant Leaf Disease Detection/`**
   - **Origin**: User-curated high-resolution dataset.
   - **Subsets**:
     - `Original DataSet/`: 865 high-resolution raw camera images (`Aphids_Disease`: 221, `Blotch`: 238, `Healthy_Leaf`: 213, `Leaf_Spot`: 193).
     - `Augmented DataSet/`: 3,496 geometric and photometric transformed images (`Aphids_Disease`: 847, `Blotch`: 909, `Healthy_Leaf`: 821, `Leaf_Spot`: 919).
   - **Count**: 4,361 images.

### 6.2 MD5 Hash-Based Deduplication Algorithm
To prevent duplicate images across sources from leaking between training and evaluation splits:
```python
with open(img_path, "rb") as fh:
    h = hashlib.md5(fh.read()).hexdigest()
if h in seen_hashes:
    continue
seen_hashes.add(h)
```
This guarantees that exact duplicates across Parquet extractions and folder copies are discarded.

### 6.3 Consolidated Class Distribution & Stratified Splitting

| Class Index | Target Class Name | Image Count | Percentage | Primary Contributing Datasets |
|:---:|:---|:---:|:---:|:---|
| **0** | **Aphids Disease** | **1,289** | 13.2% | Dataset C (Original + Augmented) + Dataset B (HF) |
| **1** | **Dry Leaf** | **812** | 8.3% | Dataset A (Augmented) |
| **2** | **Healthy Leaf** | **2,232** | 22.8% | Dataset A + Dataset B + Dataset C |
| **3** | **Leaf Blotch** | **2,377** | 24.3% | Dataset A + Dataset B + Dataset C |
| **4** | **Leaf Spot** | **1,305** | 13.4% | Dataset C (Original + Augmented) + Dataset B (HF) |
| **5** | **Rhizome Disease Root** | **910** | 9.3% | Dataset A (Augmented) |
| **6** | **Rhizome Healthy Root** | **846** | 8.7% | Dataset A (Augmented) |
| **Total** | **Combined Dataset** | **9,771** | **100.0%** | **100% Unique, MD5-Verified** |

---

## 7. Model Evaluation, Confusion Matrix & Performance Metrics (98.43%)

### 7.1 Confusion Matrix Analysis
Evaluated across **1,466 untouched test samples** (15% stratified test split):

```text
                                         PREDICTED CLASS
                  Aphids    Dry   Healthy  Blotch    Spot   RhizomeDis RhizomeHlt   Total True
                ┌─────────────────────────────────────────────────────────────┐
Aphids Disease  │  187        0        6       0       0        0          0  │    193
Dry Leaf        │    0      122        0       0       0        0          0  │    122
Healthy Leaf    │    1        0      334       0       0        0          0  │    335
Leaf Blotch     │    0        1        3     353       0        0          0  │    357
Leaf Spot       │    4        0        0       7     185        0          0  │    196
Rhizome Dis Root│    0        0        0       0       0      136          0  │    136
Rhizome Hlt Root│    0        0        0       0       0        1        126  │    127
                └─────────────────────────────────────────────────────────────┘
Total Predictions: 1,466  |  Correct: 1,443  |  Errors: 23  |  Test Accuracy: 98.43%
```

### 7.2 Mathematical Definitions of Performance Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1,443}{1,466} = 98.43\%$$

$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}, \quad \text{Recall}_i = \frac{TP_i}{TP_i + FN_i}, \quad F_1\text{-Score}_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

### 7.3 Per-Class Precision, Recall, Specificity, and F1-Scores

| Class Name | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Aphids Disease** | 187 | 5 | 6 | **97.4%** | **96.9%** | **0.97** |
| **Dry Leaf** | 122 | 1 | 0 | **99.2%** | **100.0%** | **1.00** |
| **Healthy Leaf** | 334 | 9 | 1 | **97.4%** | **99.7%** | **0.99** |
| **Leaf Blotch** | 353 | 7 | 4 | **98.1%** | **98.9%** | **0.98** |
| **Leaf Spot** | 185 | 0 | 11 | **100.0%** | **94.4%** | **0.97** |
| **Rhizome Disease Root** | 136 | 1 | 0 | **99.3%** | **100.0%** | **1.00** |
| **Rhizome Healthy Root** | 126 | 0 | 1 | **100.0%** | **99.2%** | **1.00** |
| **Macro Average** | — | — | — | **98.8%** | **98.4%** | **0.986** |
| **Weighted Average** | — | — | — | **98.5%** | **98.4%** | **0.984** |

---

## 8. Computer Vision Disease Severity Quantification (OpenCV HSV)

In addition to classifying the disease, the application measures the physical area of damage on the leaf lamina using HSV color segmentation.

```text
               Input BGR Image (from user upload)
                             │
                             ▼
               Convert to HSV Color Space [H, S, V]
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    Healthy Foliage Mask           Necrotic Spot Mask
    • Hue: 20 to 90                • Hue: 5 to 24 (Necrosis 1)
    • Sat: 30 to 255               • Hue: 0 to 180, V<=65 (Necrosis 2)
    • Val: 30 to 255               • Combined via bitwise_or
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
              Full Leaf Mask = Green | Diseased
                             │
                             ▼
              Morphological Closing (5x5 Ellipse Kernel)
                             │
                             ▼
              Spots on Leaf = Necrotic & Full Leaf
                             │
                             ▼
              Infected Area % = (Spots Pixels / Leaf Pixels) * 100
```

### 8.1 HSV Color Space Conversion & Mathematical Formulation
For an RGB pixel $(R, G, B)$ normalized to $[0, 1]$:
$$V = \max(R, G, B), \quad \Delta = V - \min(R, G, B)$$
$$H = 60^\circ \times \begin{cases}
0^\circ, & \text{if } \Delta = 0 \\
\left( \frac{G - B}{\Delta} \bmod 6 \right), & \text{if } V = R \\
\left( \frac{B - R}{\Delta} + 2 \right), & \text{if } V = G \\
\left( \frac{R - G}{\Delta} + 4 \right), & \text{if } V = B
\end{cases}, \quad S = \begin{cases} 0, & \text{if } V = 0 \\ \frac{\Delta}{V}, & \text{if } V \ne 0 \end{cases}$$

### 8.2 Dual-Mask Necrotic Spot Segmentation
Foliar spots exhibit two primary color phases:
1. **Yellow-Brown Active Necrosis**:
   $$H \in [5, 24], \quad S \in [40, 255], \quad V \in [25, 210]$$
2. **Dark Brown / Black Desiccated Lesions**:
   $$H \in [0, 180], \quad S \in [25, 255], \quad V \in [20, 65]$$

### 8.3 Morphological Closing & Contour Operations
To fill small pinhole artifacts and bridge discontinuous lesions:
$$M_{\text{closed}} = (M_{\text{leaf}} \oplus K) \ominus K$$
Using an elliptical structuring element $K$ of size $5 \times 5$:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
total_leaf = cv2.morphologyEx(total_leaf, cv2.MORPH_CLOSE, kernel)
spots_on_leaf = cv2.bitwise_and(disease_mask, total_leaf)
```

### 8.4 Clinical Severity Categorization Scale

```text
Severity Scale:
[0% ─── Healthy ─── 3%] ─────── [Mild: 3% - 10%] ─────── [Moderate: 10% - 25%] ─────── [Severe: >= 25%]
```

- **🟢 Healthy / Normal** ($< 3.0\%$): Regular foliage without necrotic symptoms.
- **🟡 Mild Damage** ($3.0\% \le \text{Area} < 10.0\%$): Isolated lesions; spot clipping and bio-fungicides recommended.
- **🟠 Moderate Damage** ($10.0\% \le \text{Area} < 25.0\%$): Noticeable leaf area destruction; foliar chemical intervention required.
- **🔴 Severe Damage** ($\ge 25.0\%$): Coalesced necrotic blighting; immediate spray schedule needed to prevent yield collapse.

---

## 9. Agronomic Compendium: Pathogens, Symptoms, Treatments & TNAU Protocols

All agronomic guidance in AgriVision is validated against published literature from the **Tamil Nadu Agricultural University (TNAU)** Agritech Portal and the **ICAR-Indian Institute of Spices Research (IISR)**.

```text
+---------------------------------------------------------------------------------------------------------+
|                                      TNAU AGRONOMIC SUMMARY TABLE                                       |
+----------------------+-----------------------+---------------------+------------------------------------+
| Common Disease Name  | தமிழ்ப் பெயர்           | Causal Pathogen     | TNAU Recommended Chemical Treatment|
+----------------------+-----------------------+---------------------+------------------------------------+
| Leaf Spot            | இலைப்புள்ளி நோய்       | Colletotrichum      | Spray Mancozeb (2 g/L) or          |
|                      |                       | capsici (Fungus)    | Carbendazim (1 g/L)                |
+----------------------+-----------------------+---------------------+------------------------------------+
| Leaf Blotch          | இலைக்கருகல் நோய்      | Taphrina maculans   | Spray Copper Oxychloride (2.5 g/L) |
|                      |                       | (Fungus)            | or Mancozeb (2 g/L)                |
+----------------------+-----------------------+---------------------+------------------------------------+
| Aphids Infestation   | அசுவினி நோய் (பூச்சி)  | Aphis gossypii      | Spray 3% Neem Oil or               |
|                      |                       | (Aphids / Vector)   | Dimethoate 30 EC (2 ml/L)          |
+----------------------+-----------------------+---------------------+------------------------------------+
| Dry Leaf             | காய்ந்த இலை           | Moisture stress /   | Regulate drip irrigation; mulch    |
|                      |                       | heat scorching      | with dry turmeric leaves           |
+----------------------+-----------------------+---------------------+------------------------------------+
| Rhizome Disease Root | கிழங்கு அழுகல் நோய்    | Pythium aphanid-    | Seed rhizome dip with Trichoderma; |
|                      |                       | ermatum (Soft Rot)  | Drench with Metalaxyl-Mancozeb (2g)|
+----------------------+-----------------------+---------------------+------------------------------------+
| Healthy Turmeric     | ஆரோக்கியமான இலை       | Normal plant vigor  | Balanced NPK (30:30:60 kg/acre)    |
|                      |                       | (No pathogen)       | Weed-free border maintenance       |
+----------------------+-----------------------+---------------------+------------------------------------+
```

### 9.1 Leaf Spot (*Colletotrichum capsici*) / இலைப்புள்ளி நோய்
* **Etiology**: Ascomycete fungus forming dark acervuli bearing curved falcate conidia; spreads via rain splash and overhead irrigation during warm, humid weather ($28^\circ\text{C}-32^\circ\text{C}$, $\text{RH}>80\%$).
* **Symptoms**:
  - Small, oval to irregular brown spots appearing on the upper surface of the leaf lamina.
  - Spots gradually enlarge and develop a prominent, bright yellow chlorotic halo.
  - Lesions coalesce along leaf margins, causing extensive scorching and premature leaf drying.
* **TNAU Management Protocol**:
  1. *Sanitation*: Collect and burn infected leaves to eliminate inoculum reservoirs.
  2. *Chemical Spray*: Apply **Mancozeb @ 2 g/L** or **Carbendazim @ 1 g/L** at first symptom onset. Repeat after 15 days if rain persists.
  3. *Drainage*: Improve furrow drainage to lower humidity in the plant canopy.
* **Yield Loss Impact**: **15% to 25%** loss in finger rhizome development.
* **தமிழ் விளக்கம்**:
  - **நோய் காரணி**: மழை நீர் மற்றும் காற்றின் மூலம் பரவும் பூஞ்சான் நோய்.
  - **அறிகுறிகள்**: இலையின் மேல் தோன்றும் பழுப்பு நிறப் புள்ளிகள், மஞ்சள் நிற வளையம், இலை விளிம்புகள் கருகி உலர்தல்.
  - **தீர்வுகள்**: மான்கோசெப் (2 கிராம்/லிட்டர்) அல்லது கார்பென்டாசிம் (1 கிராம்/லிட்டர்) தெளிக்கவும்.

### 9.2 Leaf Blotch (*Taphrina maculans*) / இலைக்கருகல் நோய்
* **Etiology**: Primitive ascomycetous fungus producing naked asci on the leaf surface; favored by overcast skies, continuous drizzle, and cooler temperatures ($21^\circ\text{C}-25^\circ\text{C}$).
* **Symptoms**:
  - Scores of tiny, circular to rectangular orange-yellow or reddish-brown spots appear on both sides of the leaf.
  - Spots coalesce into irregular, dirty-yellow necrotic patches.
  - Leaves turn brittle, curl upward along the midrib, and dry prematurely.
* **TNAU Management Protocol**:
  1. *Foliar Spray*: Spray **Copper Oxychloride (COC) @ 2.5 g/L** or **Mancozeb @ 2 g/L** at early spot formation.
  2. *Balanced Nutrition*: Avoid excessive top-dressing with Urea/Nitrogen fertilizers, which produces soft, susceptible leaves.
* **Yield Loss Impact**: **10% to 20%** reduction in total yield.
* **தமிழ் விளக்கம்**:
  - **நோய் காரணி**: காற்றில் பரவும் பூஞ்சான்; குளிர்ந்த பனி மற்றும் மேகமூட்டமான காலநிலையில் வேகமாகப் பரவுகிறது.
  - **அறிகுறிகள்**: இலையின் இருபுறமும் தோன்றும் சிறிய செம்பழுப்பு நிறப் புள்ளிகள், இலைகள் கருகி முன்கூட்டியே உதிர்தல்.
  - **தீர்வுகள்**: காப்பர் ஆக்ஸிகுளோரைடு (2.5 கிராம்/லிட்டர்) அல்லது மான்கோசெப் (2 கிராம்/லிட்டர்) தெளிக்கவும்.

### 9.3 Aphids Infestation (*Aphis gossypii*) / அசுவினி நோய்
* **Etiology**: Small, soft-bodied sap-sucking insects (aphids) that colonize the underside of leaves and tender shoots, excreting honeydew.
* **Symptoms**:
  - Chlorotic yellowing and crinkling of leaf blades.
  - Sooty mold (black superficial fungal coating) growing on aphid honeydew, obstructing photosynthesis.
* **TNAU Management Protocol**:
  1. *Organic Intervention*: Spray **3% Neem Oil (Azadirachtin)** with an emulsifier.
  2. *Insecticidal Spray*: In severe outbreaks, spray **Dimethoate 30 EC @ 2 ml/L** or **Methyl Demeton 25 EC @ 2 ml/L**.
* **தமிழ் விளக்கம்**:
  - **அறிகுறிகள்**: இலைகளின் சாற்றை உறிஞ்சும் பூச்சிகள், இலைகள் சுருங்குதல், கரும்பூஞ்சாணம் படர்தல்.
  - **தீர்வுகள்**: வேப்பெண்ணெய் 3% அல்லது டைமெத்தோயேட் (2 மி.லி/லிட்டர்) தெளிக்கவும்.

### 9.4 Dry Leaf / காய்ந்த இலை
* **Etiology**: Non-parasitic abiotic condition caused by prolonged dry spells, low soil moisture, high vapor pressure deficits, and scorching solar radiation.
* **Symptoms**: Leaf tips and margins turn straw-colored and dry down the leaf blade without distinct fungal halos.
* **Corrective Measures**: Regulate furrow or drip irrigation cycles; apply paddy straw or coir pith mulching to conserve root-zone moisture.
* **தமிழ் விளக்கம்**: நீர்ப்பற்றாக்குறை மற்றும் கடுமையான வெயில் காரணமாக இலைகள் காய்ந்து போதல்; சீரான பாசனம் மற்றும் தழைக்கூளம் இடவும்.

### 9.5 Rhizome Rot (*Pythium aphanidermatum*) / கிழங்கு அழுகல் நோய்
* **Etiology**: Soil-borne oomycete water mold causing collar rot and soft rot in waterlogged fields.
* **Symptoms**: Foliage shows progressive yellowing from the lower leaves upwards; the pseudostem base becomes soft and water-soaked, emitting a foul odor.
* **Management**: Dipping seed rhizomes in *Trichoderma viride* ($4\text{ g/kg}$) before planting; drenching affected clumps with **Metalaxyl-Mancozeb @ 2 g/L**.
* **தமிழ் விளக்கம்**: அதிக நீர் தேங்குவதால் ஏற்படும் கிழங்கு அழுகல் நோய்; விதை நேர்த்தி மற்றும் முறையான வடிகால் வசதி செய்தல்.

### 9.6 Healthy Turmeric Foliage / ஆரோக்கியமான இலை
* **Vigor Profile**: Broad, upright, deep-green lamina with clean margins and no fungal lesions or chlorotic halos.
* **Maintenance**: Apply balanced NPK fertilizers ($30:30:60\text{ kg/acre}$ basal, with split top-dressings), perform regular field scouting, and maintain clean field borders.
* **தமிழ் விளக்கம்**: அடர்பச்சை நிறமுடைய ஆரோக்கியமான இலைகள்; முறையான உரமிடுதல் மற்றும் கள பராமரிப்பு தொடரவும்.

---

## 10. Dual-Engine Agricultural Advisory & Sub-Millisecond Caching

The advisory subsystem (`api_client.py`) combines LLM intelligence with a local, verified offline database:

```text
                        Disease Classification Request
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Step 1: Check Local Cache │
                        │      info_cache.json      │
                        └─────────────┬─────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
              [Found in Cache]                     [Cache Miss]
                   ▼                                     ▼
      ┌─────────────────────────┐           ┌─────────────────────────┐
      │ Return Cached Data      │           │ Step 2: Check API_KEY   │
      │ Latency: < 1 ms         │           │ in Environment (.env)   │
      └─────────────────────────┘           └────────────┬────────────┘
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                             [Key Exists]                                [Key Missing]
                                  ▼                                             ▼
                    ┌───────────────────────────┐                 ┌───────────────────────────┐
                    │ Step 3A: External API     │                 │ Step 3B: Offline Fallback │
                    │ • Google Gemini 1.5 Flash │                 │ Query diseases.json       │
                    │ • OpenAI GPT-4o-mini      │                 │ Latency: < 2 ms           │
                    └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                                  │                                             │
                       ┌──────────┴──────────┐                                  │
                   [Success]              [Failure]                             │
                       ▼                     ▼                                  │
          ┌────────────────────────┐  ┌───────────────┐                         │
          │ Save to info_cache.json│  │ Fallback to   │                         │
          │ Return Live Data       │  │ diseases.json │                         │
          └────────────────────────┘  └───────┬───────┘                         │
                                              │                                 │
                                              └────────────────┬────────────────┘
                                                               │
                                                               ▼
                                                  Save to info_cache.json
                                                  Return Verified Data
```

### 10.1 Structured Anti-Hallucination System Prompting
When querying external APIs, the prompt enforces strict boundaries to avoid hallucinations:
1. **Source Grounding**: Output must rely exclusively on TNAU, ICAR, and State Agricultural University materials.
2. **No Invented Dosages**: If an exact dosage is absent, the model must output *"check the product label or ask your agriculture officer"*.
3. **Simultaneous Bilingual Payloads**: Produces both English and Tamil advice in a single JSON call:
   ```json
   {
     "scientific_name": "Colletotrichum capsici",
     "cause": "Fungal pathogen favored by high humidity and rain splash.",
     "symptoms": ["Brown spots with yellow halos on leaf surface"],
     "solution": ["1. Collect and burn infected leaves.", "2. Spray Mancozeb @ 2 g/L."],
     "prevention": ["Crop rotation with non-host cereals"],
     "yield_loss": "15% - 25% (TNAU Agritech Portal)",
     "tamil": {
       "disease_name": "இலைப்புள்ளி நோய் (Leaf Spot)",
       "solution": ["1. பாதிக்கப்பட்ட இலைகளை அகற்றி எரிக்கவும்.", "2. மான்கோசெப் (2 கிராம்/லிட்டர்) தெளிக்கவும்."]
     }
   }
   ```

### 10.2 Multi-Provider Support (Google Gemini & OpenAI)
- **Google Gemini**: Uses the `gemini-1.5-flash` model with `response_mime_type: "application/json"`.
- **OpenAI**: Triggered automatically if `API_KEY` starts with `sk-`, using `gpt-4o-mini` with `response_format: {"type": "json_object"}`.

### 10.3 High-Speed Local Caching Architecture (`info_cache.json`)
Every unique disease query is cached locally. Repeated scans for the same condition return in **$< 1\text{ ms}$**, eliminating network latency and conserving API quotas.

### 10.4 Zero-Downtime Offline Fallback Database (`diseases.json`)
If the API key is not configured, the network times out, or the API returns an error, AgriVision falls back to `diseases.json` without interrupting the user experience. The UI displays an advisory badge: *"Offline information from verified TNAU advisory database"*.

---

## 11. Frontend User Experience & Interactive Design (Google Stitch)

### 11.1 Agricultural Design System & Palette Tokens

```css
:root {
  --primary-gold:   #C98A0B; /* Turmeric curcumin gold */
  --foliage-green:  #2F6B3A; /* Leaf chloroplastic green */
  --cream-bg:       #FAF6EA; /* Warm, high-contrast outdoor parchment */
  --dark-text:      #23301F; /* Deep botanical dark forest */
  --alert-red:      #B93828; /* Necrotic severity and warning indicator */
  --card-bg:        #FFFFFF; /* Elevated clean panel surface */
}
```

### 11.2 Native Camera & Drag-and-Drop Dropzone
- **Desktop**: Intuitive drag-and-drop zone with drag-over visual feedback.
- **Mobile**: Uses native OS camera capture via `<input type="file" accept="image/*" capture="environment">`.

### 11.3 Animated Cybernetic Laser Scanline
During inference, a golden laser line sweeps vertically across the leaf preview using CSS keyframe animations, providing clear visual feedback that computer vision analysis is underway.

### 11.4 Dynamic SVG Circular Progress Ring
Visualizes model confidence using SVG circle stroke-dashoffset animation:
$$\text{Offset} = \text{Circumference} \times \left(1 - \frac{\text{Confidence}}{100}\right)$$

### 11.5 Actionable Accordions & Treatment Steps
Numbered chemical and cultural treatment solutions are **expanded by default**, giving farmers immediate access to treatment steps without extra taps. Symptoms and prevention tips remain collapsible.

### 11.6 Instant Client-Side Bilingual Translation Engine
Clicking the **தமிழ்** button updates all UI headings, diagnostic titles, symptoms, and treatment instructions using local DOM replacement without refreshing the page.

### 11.7 Printable Diagnostic Dossier & PDF Generation (`/report/pdf`)
Sends scan results to `/report/pdf`, which compiles a branded PDF report using ReportLab with:
- High-resolution leaf photo thumbnail.
- Severity rating and affected surface percentage.
- TNAU-certified treatment and prevention steps.
- Official institutional citations and advisory notes.

### 11.8 LocalStorage Scan History Log (`/history`)
The application saves recent scans both in the browser's `localStorage` and in `data/history.json`, allowing farmers to review earlier diagnoses along with timestamps and severity ratings.

---

## 12. Comprehensive Codebase & File-by-File Reference

```text
mini_project/
├── app.py                      # Flask application: 3-tier validation, routing, inference, PDF export
├── api_client.py               # External API client (Gemini/OpenAI), caching, and TNAU fallback
├── download_hf_dataset.py      # Hugging Face Parquet streaming dataset extractor
├── train_model.py              # CLI training pipeline with automatic class discovery & evaluation
├── train_model.ipynb           # Interactive Jupyter notebook for model development
├── diseases.json               # Local verified TNAU database (bilingual EN & TA)
├── info_cache.json             # Persistent local JSON cache for external API responses
├── requirements.txt            # Python dependencies (TensorFlow, Flask, OpenCV, ReportLab)
├── .env.example                # Template for environment variables (API_KEY)
├── .gitignore                  # Prevents cache, virtual environments, and weights from committing
├── README.md                   # Complete technical documentation
│
├── data/
│   └── history.json            # Local JSON log storing recent inspection scans
│
├── model/
│   ├── class_names.json        # 7-class vocabulary array mapping indices to names
│   ├── turmeric_model.keras    # Full trained Keras model (23.8 MB)
│   ├── turmeric_model.tflite   # Quantized edge-optimized TFLite model (2.72 MB)
│   └── confusion_matrix.png    # Evaluation heatmap on untouched test dataset
│
├── sample_images/              # Sample images for testing and demonstration
│   ├── healthy_turmeric_leaf.jpg
│   ├── leaf_spot_turmeric.jpg
│   └── non_turmeric_sample.jpg
│
├── static/
│   ├── script.js               # Frontend JavaScript: UI state, camera, scanline, bilingual toggle
│   ├── style.css               # Design system, color tokens, dark mode, responsive layout
│   └── uploads/                # Temporary storage for inspected user images
│
├── templates/
│   ├── upload.html             # Landing upload studio with drag-and-drop & camera capture
│   ├── result.html             # Diagnostic dashboard with severity & TNAU solutions
│   ├── invalid_warning.html    # Rejection notice for non-leaf or non-turmeric uploads
│   ├── history.html            # Chronological log of past diagnostic evaluations
│   └── index.html              # Single-page alternate template
│
├── Turmeric Plant Disease Augmented Dataset/         # Dataset A (Local augmented repository)
├── dataset_hf/                                       # Dataset B (Extracted Hugging Face repository)
└── Image Dataset for Turmeric Plant Leaf Disease Detection/ # Dataset C (High-res original & augmented)
    ├── Original DataSet/
    └── Augmented DataSet/
```

---

## 13. Installation, Dependencies & Step-by-Step Execution Guide

### 13.1 Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows 10/11 (WSL2 recommended).
- **Python**: Version `3.10` or higher.
- **System Libraries**: `libgl1` and `libglib2.0-0` (required by OpenCV for headless image processing):
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```

### 13.2 Virtual Environment & Installation
```bash
# 1. Navigate to the project root
cd /home/tom/Projects/mini_project

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 13.3 Environment Setup (Optional External API)
The application works completely offline using `diseases.json`. To enable live Gemini or OpenAI summaries:
```bash
cp .env.example .env
```
Edit `.env` and add your key:
```ini
API_KEY=your_gemini_or_openai_api_key_here
```

### 13.4 Running the Application
```bash
python3 app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 14. Command-Line Interface (CLI) Training Guide

The training script [`train_model.py`](file:///home/tom/Projects/mini_project/train_model.py) automatically discovers classes across nested folder structures, removes duplicate images by MD5 hash, and balances class weights.

### 14.1 Inspecting Datasets Without Training (`--dry-run`)
Scan directories and verify image counts across all classes without allocating GPU memory or starting training:
```bash
python3 train_model.py --dry-run
```

### 14.2 Training with Default Datasets
Train MobileNetV2 on the combined 9,771-image dataset (Datasets A, B, and C):
```bash
python3 train_model.py
```

### 14.3 Supplying Additional Custom Datasets
Pass extra directories directly via the CLI:
```bash
python3 train_model.py --extra-dataset "/path/to/another/turmeric_dataset/"
```

### 14.4 Custom Hyperparameters
```bash
python3 train_model.py \
    --epochs 10 \
    --fine-tune-epochs 5 \
    --batch-size 32
```

---

## 15. REST API Specification (Endpoints, Payloads, Responses)

### 15.1 API Endpoint Reference

| Endpoint | Method | Format | Description |
|:---|:---:|:---:|:---|
| `/` | `GET` | HTML | Renders the upload studio (`upload.html`). |
| `/history` | `GET` | HTML | Displays past inspection records (`history.html`). |
| `/predict` | `POST` | Multipart Form / JSON | Uploads leaf, runs 3-tier validation, returns diagnostic analysis. |
| `/report/pdf` | `POST` | PDF Binary | Compiles and streams a formatted PDF clinical dossier. |

### 15.2 Programmatic Prediction Example (cURL)

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Accept: application/json" \
     -F "file=@sample_images/leaf_spot_turmeric.jpg"
```

**JSON Response (Sample)**:
```json
{
  "is_turmeric_leaf": true,
  "leaf_confirmed_message": "Turmeric leaf confirmed",
  "disease": "Leaf Spot",
  "confidence": 98.4,
  "severity": "Moderate",
  "scientific_name": "Colletotrichum capsici",
  "cause": "Airborne and rain-splash fungal pathogen favored by high humidity.",
  "symptoms": [
    "Small oval brown spots on leaf surface",
    "Prominent yellow chlorotic halos surrounding lesions",
    "Coalescing spots causing leaf margin scorch"
  ],
  "infected_area_pct": 14.2,
  "yield_loss": "15% - 25% (TNAU Agritech Portal)",
  "solution": [
    "1. Collect and burn infected leaves.",
    "2. Spray Mancozeb @ 2 g/L or Carbendazim @ 1 g/L.",
    "3. Improve furrow drainage to reduce microclimate humidity."
  ],
  "prevention": [
    "Treat seed rhizomes with fungicide slurry before planting",
    "Practice crop rotation with non-host crops every 2-3 years"
  ],
  "sources": [
    {
      "title": "TNAU Agritech Portal - Turmeric Disease Management",
      "url": "https://agritech.tnau.ac.in/crop_protection/turmeric_diseases.html"
    }
  ],
  "tamil": {
    "disease_name": "இலைப்புள்ளி நோய் (Leaf Spot)",
    "scientific_name": "கோலெட்டோட்ரைக்கம் கேப்சிசி (Colletotrichum capsici)",
    "solution": [
      "1. பாதிக்கப்பட்ட இலைகளை சேகரித்து எரிக்கவும்.",
      "2. மான்கோசெப் (2 கிராம்/லிட்டர்) தெளிக்கவும்."
    ]
  },
  "is_offline": false,
  "scan_id": "AV-7429",
  "scan_time": "Oct 03, 2026 • 08:30 PM"
}
```

---

## 16. Production Deployment & Edge Device Guide (Gunicorn, Nginx, TFLite on Raspberry Pi)

### 16.1 Production WSGI Server with Gunicorn
For production deployments, run the application behind **Gunicorn**:
```bash
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 --timeout 120 --access-logfile - app:app
```

### 16.2 Systemd Service Configuration
Create `/etc/systemd/system/agrivision.service`:
```ini
[Unit]
Description=AgriVision Turmeric Inspector Flask Application
After=network.target

[Service]
User=tom
WorkingDirectory=/home/tom/Projects/mini_project
Environment="PATH=/home/tom/Projects/mini_project/.venv/bin"
ExecStart=/home/tom/Projects/mini_project/.venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agrivision
sudo systemctl start agrivision
```

### 16.3 Edge Deployment on Raspberry Pi (TFLite Runtime)
The quantized model (`model/turmeric_model.tflite`, 2.72 MB) can run on a Raspberry Pi using the lightweight `tflite-runtime` library (no full TensorFlow installation needed):
```bash
pip install tflite-runtime Pillow numpy
```

```python
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model/turmeric_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("sample_images/leaf_spot_turmeric.jpg").convert("RGB").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]
```

---

## 17. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: Why did my upload get rejected with "Insufficient leaf foliage pixels detected"?
**Answer**: Tier 2 validation checks for green, yellow, or brown leaf pixels via HSV segmentation. If the image is blurry, out of focus, or dominated by soil, walls, or hands ($< 5\%$ plant pixels), it will be rejected. Ensure the photo clearly captures the leaf blade against a neutral background under natural daylight.

### Q2: Why does the terminal show CUDA driver warnings on my machine?
**Answer**: TensorFlow checks for an NVIDIA GPU by default:
```text
Could not find cuda drivers on your machine, GPU will not be used.
```
This is purely informational. The application will seamlessly use all available CPU cores with oneDNN AVX2 optimizations.

### Q3: How do I operate the application with zero internet connectivity?
**Answer**: The system runs fully offline by default. If `API_KEY` is not present in `.env`, the app serves verified agricultural recommendations directly from `diseases.json`.

### Q4: Can I train on a custom folder of turmeric leaves?
**Answer**: Yes. Point the training script to your image folder using the `--extra-dataset` flag:
```bash
python3 train_model.py --extra-dataset "/path/to/my_dataset/"
```
The script will recursively detect and normalize class folder names automatically.

---

## 18. Academic Citations, References & Official Disclaimer

### Institutional References
1. **Tamil Nadu Agricultural University (TNAU)**:
   - *TNAU Agritech Portal*: Crop Protection Compendium for Spices & Condiments (*Curcuma longa* L.).
   - Department of Plant Pathology, Center for Plant Protection Studies, AC&RI, Coimbatore, Tamil Nadu.
2. **ICAR - Indian Institute of Spices Research (IISR)**:
   - *Major Diseases of Turmeric and their Management*, Extension Pamphlet Series, Calicut, Kerala.
3. **Krishi Vigyan Kendra (KVK) - Erode & Perundurai**:
   - Seasonal field advisories on Rhizome Rot and Foliar Blotch incidence in the lower Bhavani project command area.

### Academic Disclaimer
> **⚠️ Agricultural Advisory Notice**:
> *AgriVision Turmeric Inspector is developed as an academic mini-project and assistive decision-support tool. Diagnosis results and chemical dosages are provided for informational guidance. Before applying chemical treatments or purchasing pesticides, farmers are strongly advised to consult their local Assistant Agricultural Officer (AAO), Horticultural Officer (HO), or TNAU Krishi Vigyan Kendra extension specialist.*
