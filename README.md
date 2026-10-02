# AI-Based Product Review Contradiction Detection

## Task 1 – AI Problem Design

### 1. Problem Statement

Online products receive thousands of customer reviews describing different experiences with product features such as battery life, camera quality, performance, price, and build quality.

When reviews contain conflicting opinions about the same product aspect, it can be difficult for consumers and product teams to identify these contradictions manually.

This project proposes an AI-based system that detects **contradictory customer opinions about the same product aspect** from a collection of product reviews.

### Example

**Review 1:**
> "The battery easily lasts for two days."

**Review 2:**
> "The battery barely lasts five hours."

**AI Output:**

```text
Aspect: Battery
Review 1: Positive
Review 2: Negative

Contradiction: Detected
```

The system identifies the contradiction but does not determine which customer's statement is factually correct.

---

## 2. Target Users

The system is intended for:

### Consumers
Consumers can identify product aspects where customer experiences vary significantly before making a purchase decision.

### Product Managers
Product teams can identify features receiving conflicting customer feedback and investigate potential product issues.

### E-commerce Analysts
Analysts can summarize large volumes of customer feedback and identify areas requiring further investigation.

For this project, the primary focus is on **consumers and product analysts**.

---

## 3. Data Source

The project will use publicly available product review data, primarily from the **Amazon Product Reviews dataset**.

The initial prototype will use a small subset of the dataset rather than processing the complete collection.

### Data Fields

| Field | Description |
|---|---|
| Product ID | Identifier of the product |
| Review Text | Written customer review |
| Rating | Customer rating |
| Aspect | Product feature discussed |
| Sentiment | Positive, Negative, or Neutral |

The `Aspect` and `Sentiment` fields may be generated or manually annotated for the selected subset.

### Example

| Review | Aspect | Sentiment |
|---|---|---|
| Battery lasts all day. | Battery | Positive |
| Battery drains very quickly. | Battery | Negative |
| The camera takes excellent photos. | Camera | Positive |

---

## 4. AI Approach

The system will use Natural Language Processing (NLP) techniques.

### Processing Pipeline

```text
Product Reviews
       ↓
Text Preprocessing
       ↓
Aspect Extraction
       ↓
Sentiment Analysis
       ↓
Group Reviews by Aspect
       ↓
Compare Opposing Opinions
       ↓
Contradiction Detection
```

The system will identify reviews discussing the same product aspect and determine whether they express opposing opinions.

For example:

```text
Battery
   │
   ├── "Battery lasts all day." → Positive
   │
   └── "Battery dies within hours." → Negative
                     ↓
             Contradiction
```

---

## 5. Constraints

The initial version of the project will operate under the following constraints:

- Only English-language reviews will be considered.
- The prototype will use approximately **500–1,000 reviews**.
- The initial system will focus on **3–5 predefined product aspects**.
- Only text-based reviews will be analyzed.
- Images, videos, and other multimedia reviews are outside the scope.
- Contradiction detection will be limited to opposing opinions about the **same product aspect**.
- The system will identify conflicting opinions but will not determine which review is factually correct.
- The system is intended as an analytical tool and should not be treated as a source of verified product facts.

---

## 6. Evaluation Approach

The system will be evaluated separately for its major components.

### Aspect Extraction

The extracted product aspects will be compared against manually labeled test data.

Metrics:

- Precision
- Recall
- F1-score

### Sentiment Classification

The predicted sentiment will be compared with manually labeled sentiment.

Metrics:

- Accuracy
- Precision
- Recall
- F1-score

### Contradiction Detection

Review pairs will be labeled as:

```text
1 → Contradictory
0 → Non-contradictory
```

The model's predictions will then be compared against the labeled test set.

Metrics:

- Precision
- Recall
- F1-score
- Confusion Matrix

### Dataset Split

The dataset will be divided into:

```text
Training Set       → 80%
Validation Set     → 10%
Testing Set        → 10%
```

The final evaluation will be performed only on the held-out test set.

---

## 7. Success Criteria

The initial prototype will aim to achieve:

- **F1-score ≥ 0.80** for sentiment classification.
- **F1-score ≥ 0.75** for contradiction detection.
- Correct identification of the product aspect associated with the conflicting opinions.
- Low false-positive rate so that ordinary differences in reviews are not unnecessarily classified as contradictions.

The F1-score is emphasized because both false positives and false negatives are important in contradiction detection.

---

## 8. Expected Output

For a given product, the system should provide an aspect-level summary.

Example:

```text
Product: XYZ Smartphone

Aspect: Battery

Positive:
"Battery easily lasts two days."

Negative:
"Battery barely lasts five hours."

Status:
Contradictory Opinions Detected

Confidence:
89%
```

The system may also provide an overall aspect summary:

```text
Aspect          Positive    Negative    Status
------------------------------------------------
Battery           62%         38%       Mixed
Camera            81%         19%       Positive
Display           76%         24%       Positive
Performance       54%         46%       Mixed
```

---

## 9. Scope

### Included

- Product review text analysis
- Product aspect identification
- Sentiment analysis
- Contradictory opinion detection
- Aspect-level review summaries

### Not Included

- Determining whether a review is factually true
- Fake review detection
- Product recommendation
- Image/video review analysis
- Real-time e-commerce integration

---

## 10. Future Development

The project can later be extended with:

- Transformer-based NLP models
- Semantic similarity using sentence embeddings
- Explainable contradiction detection
- Product comparison
- Interactive visualization
- Review trend analysis
- Support for multiple languages
- Integration with e-commerce review APIs

---

## Conclusion

The proposed system addresses the problem of identifying conflicting customer opinions within large collections of product reviews.

By combining **aspect extraction, sentiment analysis, and contradiction detection**, the system can transform unstructured customer reviews into structured insights that are easier for consumers and product teams to understand.

---

# Task 2 – Model Integration Prototype

## 1. Overview

Task 2 implements a working prototype that integrates an AI sentence-transformer model into the contradiction detection pipeline defined in Task 1.

The prototype loads real Amazon electronics customer reviews, cleans the text, maps star ratings to sentiment labels, generates semantic embeddings using a lightweight local model, calculates cosine similarity between reviews of the same product, and flags contradictory review pairs.

> **Important Note:** This is an exploratory prototype operating on **semantic similarity + sentiment polarity**, not a fact-verification system. It flags reviews that discuss related product aspects with conflicting positive and negative sentiments.

---

## 2. Dataset Information

- **Dataset**: Amazon Reviews 2023 – Electronics (sampled subset)
- **Local File**: `data/train-00000-of-00001.parquet` (7,109 reviews)
- **Source**: [McAuley-Lab / Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Required Fields**:
  - `parent_asin`: Unique product identifier used to group reviews for the same product
  - `product_title`: Human-readable name of the product
  - `rating`: Customer star rating (1.0 to 5.0)
  - `review_text`: Unstructured text of the customer review

### Git & Storage Policy
The `.parquet` file is excluded from git commits via `.gitignore` (`data/*.parquet`) to keep the repository lightweight. If cloning fresh, place the dataset parquet in the `data/` folder.

---

## 3. Model Architecture

- **Model**: `sentence-transformers/paraphrase-MiniLM-L3-v2`
- **Embedding Dimension**: 384 dimensions
- **Model Size**: ~17 MB (extremely lightweight, runs smoothly on CPU)
- **Frameworks**: `sentence-transformers`, `scikit-learn`, `pandas`, `pyarrow`, `numpy`
- **Hardware Requirement**: Standard CPU (no GPU, vector DB, or paid LLM APIs needed)
- **Security**: No API keys or tokens required; runs completely offline once weights are cached locally.

---

## 4. Contradiction Detection Pipeline

```
[Raw Parquet Dataset] (7,109 reviews)
          │
          ▼
[Preprocessing & Cleaning] (Drop nulls, strip whitespace, remove invisible characters)
          │
          ▼
[Sentiment Labeling]
  • 1.0 – 2.0 Stars  →  Negative
  • 3.0 Stars        →  Neutral
  • 4.0 – 5.0 Stars  →  Positive
          │
          ▼
[MiniLM Embeddings] (384-dimensional normalized sentence vectors)
          │
          ▼
[Product Grouping] (Group reviews by parent_asin)
          │
          ▼
[Pairwise Cosine Similarity]
          │
          ▼
[Contradiction Filter Rule]
  ✓ Same product (identical parent_asin)
  ✓ High semantic similarity (cosine similarity ≥ threshold, default 0.60)
  ✓ Opposite sentiment polarity (Positive ↔ Negative)
          │
          ▼
[Ranked Contradictory Pairs] (examples/example_output.json & terminal report)
```

---

## 5. Project Directory Structure

```
AI-Based-Product-Review-Contradiction-Detection/
├── data/
│   └── train-00000-of-00001.parquet    # Local Parquet dataset (gitignored)
├── src/
│   ├── __init__.py                     # Package marker
│   ├── load_data.py                    # Dataset loading & column validation
│   ├── preprocess.py                   # Cleaning, text normalization & sentiment mapping
│   ├── similarity.py                   # MiniLM model loading & cosine similarity
│   └── contradiction.py                # Contradiction detection & formatting
├── examples/
│   └── example_output.json             # Saved contradictory pairs output
├── main.py                             # CLI entry point for running prototype
├── requirements.txt                    # Minimal Python dependencies
├── .gitignore                          # Ignores large datasets, caches, envs
└── README.md                           # Comprehensive documentation (Tasks 1 & 2)
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.9 or higher

### 1. Clone the repository
```bash
git clone https://github.com/Sambram123/AI-Based-Product-Review-Contradiction-Detection.git
cd AI-Based-Product-Review-Contradiction-Detection
```

### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Running the Prototype

Run the pipeline from the project root:

```bash
python main.py
```

### CLI Options:
```bash
# Set custom similarity threshold and output count
python main.py --threshold 0.65 --top-n 10

# Save output to examples/example_output.json
python main.py --threshold 0.60 --top-n 5 --save
```

---

## 8. Example Inputs & Detected Outputs

The following real examples were detected from the Amazon Electronics dataset:

### Example 1: Logitech M510 Wireless Mouse (ASIN: `B003NR57BY`)
- **Review 1 (Positive, 5 Stars)**:
  > *"This mouse works very well and I haven't had to change batteries since I bought it 3 months ago, and I never turn it off. The mouse does like a mouse surface, forget about a glass table top. Works on your leg just fine, bed, chair arm, etc.."*
- **Review 2 (Negative, 1 Star)**:
  > *"Bought this mouse, it worked great for about 2 months.. then it just died. Replaced batteries and all. When I tried to reinstall it, my laptop tells me it doesn't recognize the device. this is the 3rd Logitech mouse I have tried, and the 3rd time it's died. I'm switching to a different brand."*
- **Cosine Similarity**: `0.7657`
- **Result**: `Contradiction Detected` (Contradictory experiences regarding battery lifespan and durability)

---

### Example 2: EBL Rechargeable Batteries & Charger (ASIN: `B0BS6KLLT1`)
- **Review 1 (Positive, 5 Stars)**:
  > *"UPDATE: I have been keeping the charger in use for several months, swapping out batteries as needed. Still working well, no overheating or other glitches. EXCELLENT VALUE!..."*
- **Review 2 (Negative, 2 Stars)**:
  > *"The charger works well but it's the batteries that are sub par. The fully charged batteries have a charge life of about half of a good brand throw away alkaline battery. So what's the use of a good charger that chargers batteries that need to be almost constantly recharged..."*
- **Cosine Similarity**: `0.7522`
- **Result**: `Contradiction Detected` (Conflicting claims on battery hold and value)

---

### Example 3: Bose SoundSport Free Earbuds (ASIN: `B074F3YW3R`)
- **Review 1 (Positive, 4 Stars)**:
  > *"The headphones have a solid fit and finish (at $250 they should). The case is quite nice and the headphone stay in place via magnets. Pros: Easy setup with the Bose App. Headphones fit snugly for long periods of time without causing ear fatigue. Deep bass... If you have $250 to spare and are in the market for very good Bluetooth ear buds these should be top of your list."*
- **Review 2 (Negative, 2 Stars)**:
  > *"I found Bose's latest wireless headphones to be a fairly serious disappointment, which was surprising and frustrating after having bought 5 or 6 pairs of headphones over the years, along with speakers and a Wave Radio... They are very big and jut from my ears in a way that looks comical... they don't pair properly with Windows 10..."*
- **Cosine Similarity**: `0.7184`
- **Result**: `Contradiction Detected` (Conflicting assessments on fit, usability, and satisfaction)
