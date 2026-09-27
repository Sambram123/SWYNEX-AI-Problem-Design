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
