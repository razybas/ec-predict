# BUILD_REPORT: EC-Predict (One-Night Build)

**Date**: 2026-10-28  
**Author / Builder**: Antigravity  
**Auditor**: Muse  
**Owner**: Yazur  
**Repository**: `razybas/ec-predict` (main)

---

## 1. Amendment Status
> [!IMPORTANT]
> **Two-Stage Design Amendment Applied**:
> - **Stage A**: Binary classification (Enzyme vs. Non-Enzyme), incorporating reviewed UniProt entries with NO EC number (`reviewed:true NOT (ec:*)`), capped at the post-deduplication enzyme total ($N = 6,622$).
> - **Stage B**: 6-Class EC classification (EC 1–6) trained strictly on enzyme rows.
> - **Headline Metric**: End-to-End 7-Way hierarchical evaluation (0 = Non-Enzyme, 1..6 = EC 1..6). Test instances are scored correct iff Stage A routes correctly AND Stage B predicts the correct EC class for enzymes.
> - **Confusion Matrix**: 7-way end-to-end confusion matrix for the MLP model on the held-out test set (`results/confusion_matrix.png`).
> - **Both models reported**: Logistic Regression baseline and MLP neural network reported across Stage A, Stage B, and End-to-End.

---

## 2. Dataset Counts & Deduplication

Data harvested from UniProtKB / Swiss-Prot via REST API streaming on 2026-10-28. Deduplicated by exact sequence globally.

| Category | Query | Fetched | Post-Dedup | Train (80%) | Test (20%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Non-Enzyme (Class 0) | `reviewed:true NOT (ec:*)` | 6,657 | 6,622 | 5,297 | 1,324 |
| EC 1 (Oxidoreductases) | `reviewed:true AND ec:1.*` | 1,200 | 1,194 | 955 | 239 |
| EC 2 (Transferases) | `reviewed:true AND ec:2.*` | 1,200 | 1,127 | 902 | 225 |
| EC 3 (Hydrolases) | `reviewed:true AND ec:3.*` | 1,200 | 1,068 | 854 | 214 |
| EC 4 (Lyases) | `reviewed:true AND ec:4.*` | 1,200 | 1,048 | 838 | 210 |
| EC 5 (Isomerases) | `reviewed:true AND ec:5.*` | 1,200 | 1,049 | 839 | 210 |
| EC 6 (Ligases) | `reviewed:true AND ec:6.*` | 1,200 | 1,136 | 909 | 227 |
| **Total Enzymes (EC 1-6)** | | **7,200** | **6,622** | **5,297** | **1,325** |
| **Total Dataset** | | **13,857** | **13,244** | **10,595** | **2,649** |

---

## 3. Split & Feature Details

- **Split Protocol**: Stratified 80/20 train/test split with `random_state=42`, executed BEFORE any feature fitting or vectorization.
  - Overall Train Split: 10,595 sequences
  - Overall Test Split: 2,649 sequences
  - Stage A Train Set: 10,595 sequences (balanced 50% non-enzyme, 50% enzyme)
  - Stage B Train Set: 5,297 sequences (enzymes only, classes 1–6)
- **Feature Extraction**:
  - Amino-acid character 3-mers (triplets) with TF-IDF weighting.
  - Capped at max 20,000 features (actual observed vocabulary: 8,226 features).
  - Fit **strictly on the 80% train split** to guarantee zero data leakage.

---

## 4. Test Metrics Exactly as Printed

Evaluation performed on held-out test set ($N = 2,649$). Each model scored exactly once.

### Summary Comparison Table

| Metric | Logistic Regression Baseline | MLP Neural Network |
| :--- | :---: | :---: |
| **Stage A Accuracy (Binary)** | **0.8396** | 0.8297 |
| **Stage A Macro-F1** | **0.8395** | 0.8297 |
| **Stage B Accuracy (6-Class on True Enzymes)** | **0.6430** | 0.6204 |
| **Stage B Macro-F1** | **0.6407** | 0.6216 |
| **End-to-End Accuracy (7-Way)** | **0.7022** | 0.6848 |
| **End-to-End Macro-F1** | **0.5939** | 0.5836 |

---

### Model 1: Logistic Regression Baseline (`max_iter=1000, C=4.0`)
```
Stage A (Binary Enzyme vs Non-Enzyme):
  Accuracy: 0.8396
  Macro-F1: 0.8395

Stage B (6-Class EC 1-6 on True Enzymes):
  Accuracy: 0.6430
  Macro-F1: 0.6407

End-to-End (7-Way Hierarchical):
  Accuracy: 0.7022
  Macro-F1: 0.5939

Classification Report (End-to-End 7-Way):
                        precision    recall  f1-score   support

            Non-Enzyme     0.8322    0.8505    0.8412      1324
EC 1 (Oxidoreductases)     0.5468    0.6360    0.5880       239
   EC 2 (Transferases)     0.4564    0.3956    0.4238       225
     EC 3 (Hydrolases)     0.4167    0.3505    0.3807       214
         EC 4 (Lyases)     0.5472    0.5524    0.5498       210
     EC 5 (Isomerases)     0.6085    0.5476    0.5764       210
        EC 6 (Ligases)     0.7727    0.8238    0.7974       227

              accuracy                         0.7022      2649
             macro avg     0.5972    0.5937    0.5939      2649
          weighted avg     0.6955    0.7022    0.6979      2649
```

---

### Model 2: MLP Neural Network (`hidden_layer_sizes=(256, 128), early_stopping=True`)
```
Stage A (Binary Enzyme vs Non-Enzyme):
  Accuracy: 0.8297
  Macro-F1: 0.8297

Stage B (6-Class EC 1-6 on True Enzymes):
  Accuracy: 0.6204
  Macro-F1: 0.6216

End-to-End (7-Way Hierarchical):
  Accuracy: 0.6848
  Macro-F1: 0.5836

Classification Report (End-to-End 7-Way):
                        precision    recall  f1-score   support

            Non-Enzyme     0.8279    0.8323    0.8301      1324
EC 1 (Oxidoreductases)     0.5738    0.5690    0.5714       239
   EC 2 (Transferases)     0.4062    0.4044    0.4053       225
     EC 3 (Hydrolases)     0.3154    0.3832    0.3460       214
         EC 4 (Lyases)     0.6117    0.5476    0.5779       210
     EC 5 (Isomerases)     0.5909    0.5571    0.5735       210
        EC 6 (Ligases)     0.8104    0.7533    0.7808       227

              accuracy                         0.6848      2649
             macro avg     0.5909    0.5782    0.5836      2649
          weighted avg     0.6904    0.6848    0.6870      2649
```

---

## 5. Runtime Breakdown

- **Total Execution Time**: 201.66s (~3.36 minutes)
- **Data Load / Verification**: 0.42s
- **Stratified Split & TF-IDF Fit**: 6.27s
- **Logistic Regression Training (Stage A + B)**: 4.17s
- **MLP Training with Early Stopping (Stage A + B)**: 190.80s
- **Evaluation & Metric Writing**: ~0.8s

---

## 6. Software & Library Versions

- **OS**: Windows 11 (AMD64)
- **Python**: 3.14.5 (tags/v3.14.5:5607950, MSC v.1944 64 bit)
- **scikit-learn**: 1.9.0
- **numpy**: 2.5.2
- **scipy**: 1.18.1
- **pandas**: 3.0.5
- **matplotlib**: 3.11.1
- **joblib**: 1.5.3
- **requests**: 2.34.2

---

## 7. Git & Commit Information

- **Repository**: `https://github.com/razybas/ec-predict`
- **Branch**: `main`
- **Visibility**: Public
- **Initial Commit SHA**: `992657e5fe33ff5bcb725432ac4fc031dddc3697`
- **Release Tree State**: Synced with GitHub remote `origin/main`

---

## 8. Blockers

**None**. All requirements and acceptance criteria satisfied end-to-end.
